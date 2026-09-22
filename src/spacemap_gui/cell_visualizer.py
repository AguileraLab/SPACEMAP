import os
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree
import scipy.ndimage as ndi

# Configure logging
logger = logging.getLogger(__name__)

try:
    import zarr
except ImportError:
    zarr = None

class CellVisualizer:
    def __init__(self, roi_dir: str | Path, seg_dir: str | Path):
        """
        Initialize the CellVisualizer with directories for ROI images and Segmentation files.
        """
        self.roi_dir = Path(roi_dir)
        self.seg_dir = Path(seg_dir)
        self.file_map = {}  # Stores matched triads: {'batch_id': {'roi': path, 'geojson': path}}
        self._metadata_cache = {} # Cache parsed metadata per batch_id: {'batch_id': {'colors': [], 'names': []}}
        self._segmentation_cache = {}
        # Open TIFF handles and per-page zarr views, reused across cells.
        # Opening one of these images is expensive: a 56000x34500 image with
        # 256x256 tiles across 38 channels has over a million tile offsets to
        # parse, which measured about 350 ms per cell when the file was
        # reopened for each one. Reusing the handle drops that to about 58 ms.
        self._tiff_handles = {}      # batch_id -> (TiffFile, mtime)
        self._page_stores = {}       # (batch_id, page_index) -> zarr array

    def _get_tiff(self, batch_id: str, roi_path) -> "tifffile.TiffFile":
        """Return an open TIFF for *batch_id*, reopening if the file changed."""
        path = Path(roi_path)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = None

        cached = self._tiff_handles.get(batch_id)
        if cached is not None:
            handle, cached_mtime = cached
            if cached_mtime == mtime:
                return handle
            self._close_batch(batch_id)

        handle = tifffile.TiffFile(path)
        self._tiff_handles[batch_id] = (handle, mtime)
        return handle

    def _get_page_store(self, batch_id: str, tif: "tifffile.TiffFile", page_index: int):
        """Return a cached zarr view of one page, or None when unavailable."""
        if zarr is None:
            return None
        key = (batch_id, page_index)
        store = self._page_stores.get(key)
        if store is not None:
            return store
        try:
            store = zarr.open(tif.pages[page_index].aszarr(), mode='r')
        except Exception as exc:
            logger.debug(f"No chunked access for {batch_id} page {page_index}: {exc}")
            return None
        self._page_stores[key] = store
        return store

    def _close_batch(self, batch_id: str) -> None:
        """Drop cached handles for one batch."""
        for key in [k for k in self._page_stores if k[0] == batch_id]:
            self._page_stores.pop(key, None)
        cached = self._tiff_handles.pop(batch_id, None)
        if cached is not None:
            try:
                cached[0].close()
            except Exception:
                pass

    def close(self) -> None:
        """Release every open image handle."""
        for batch_id in list(self._tiff_handles):
            self._close_batch(batch_id)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def match_file_triads(self, unique_batch_ids: list[str]) -> dict:
        """
        Scans roi_dir and seg_dir to find matching files for each batch_id.
        Uses prefix matching.
        """
        matches = {}
        
        if not self.roi_dir.exists() or not self.seg_dir.exists():
            logger.error(f"Directories not found: ROI={self.roi_dir}, SEG={self.seg_dir}")
            return matches

        # List all available files once
        try:
            # ROI files: check multiple extensions
            roi_files = [f for f in self.roi_dir.iterdir() if f.name.lower().endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]
            # Seg files: check json/geojson in both self.seg_dir and self.seg_dir/geojson
            seg_files = [f for f in self.seg_dir.iterdir() if f.name.lower().endswith(('.json', '.geojson'))]
            geojson_sub = self.seg_dir / "geojson"
            if geojson_sub.exists() and geojson_sub.is_dir():
                seg_files.extend([f for f in geojson_sub.iterdir() if f.name.lower().endswith(('.json', '.geojson'))])
        except Exception as e:
            logger.error(f"Error scanning directories: {e}")
            return matches

        for batch_id in unique_batch_ids:
            batch_id_str = str(batch_id)
            
            # Helper to find best match
            def find_best_match(candidates, target_id):
                best_file = None
                best_score = 0.0
                
                for f in candidates:
                    fname = f.name
                    # 1. Exact containment (Highest priority)
                    if target_id in fname:
                        # Favor shorter filenames when multiple contain the ID
                        # e.g. "P1_roi_1" matches "P1_roi_1.tif" and "P1_roi_1_copy.tif"
                        # We want the one where target_id IS the stem or close to it.
                        
                        # Simple logic: Ratio of lengths?
                        # Or better: check if it starts with target_id
                        
                        current_score = 0
                        if fname.startswith(target_id):
                            current_score += 0.5
                            
                        # Containment score
                        current_score += 0.5
                        
                        # Penalty for extra characters (length difference)
                        len_diff = abs(len(fname) - len(target_id))
                        current_score -= (len_diff * 0.01) 
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_file = f
                
                return best_file

            # Find matching ROI
            roi_match = find_best_match(roi_files, batch_id_str)
            
            # Find matching GeoJSON
            seg_match = find_best_match(seg_files, batch_id_str)

            if roi_match and seg_match:
                matches[batch_id] = {
                    'roi': str(roi_match),
                    'geojson': str(seg_match)
                }
        
        self.file_map = matches
        return matches

    def _load_geojson_index(self, geojson_path: Path) -> dict:
        """Load and cache polygon geometry plus a spatial index for a GeoJSON file."""
        cache_key = str(geojson_path)
        cached = self._segmentation_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)

            polygons = []
            if 'features' in data:
                for feature in data['features']:
                    geometry = feature.get('geometry')
                    if not geometry:
                        continue
                    polygon = shape(geometry)
                    if polygon.geom_type == 'Polygon':
                        polygons.append(polygon)
                    elif polygon.geom_type == 'MultiPolygon':
                        polygons.extend(list(polygon.geoms))

            index = {
                'polygons': polygons,
                'tree': STRtree(polygons) if polygons else None,
            }
            self._segmentation_cache[cache_key] = index
            return index
        except Exception as e:
            logger.error(f"Failed to load GeoJSON {geojson_path}: {e}")
            empty = {'polygons': [], 'tree': None}
            self._segmentation_cache[cache_key] = empty
            return empty

    def _get_cached_metadata(self, batch_id: str, tif: tifffile.TiffFile) -> dict:
        """Return cached OME channel metadata for a batch."""
        cached_meta = self._metadata_cache.get(batch_id)
        if cached_meta is not None:
            return cached_meta

        ome_metadata = tif.ome_metadata
        channel_count = len(tif.pages)
        channel_colors = []
        channel_names = []

        if ome_metadata:
            channel_colors = self._parse_ome_colors(ome_metadata)
            channel_names = self._parse_ome_channel_names(ome_metadata)

        if len(channel_colors) != channel_count:
            channel_colors = [None] * channel_count
        if len(channel_names) != channel_count:
            channel_names = [f"Ch{i}" for i in range(channel_count)]

        cached_meta = {
            'colors': channel_colors,
            'names': channel_names,
        }
        self._metadata_cache[batch_id] = cached_meta
        return cached_meta

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        normalized = str(name).strip().lower()
        for suffix in ('_norm_resolve', '_resolve', '_norm', '_corr', '_unmixed', '_log'):
            normalized = normalized.replace(suffix, '')
        return normalized.strip()

    def _resolve_channel_indices(self, requested_channels: list[str], available_channel_names: list[str]) -> list[int]:
        """Map marker names to TIFF channel indices with exact-then-fuzzy matching."""
        available_norm = [self._normalize_channel_name(name) for name in available_channel_names]
        selected_indices = []

        for requested in requested_channels:
            target = self._normalize_channel_name(requested)
            found_idx = -1

            if target in available_norm:
                found_idx = available_norm.index(target)
            else:
                for idx, channel_name in enumerate(available_norm):
                    if target and (target in channel_name or channel_name in target):
                        found_idx = idx
                        break

            if found_idx != -1 and found_idx not in selected_indices:
                selected_indices.append(found_idx)

        return selected_indices

    def _read_page_crop(
        self,
        page: tifffile.TiffPage,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        store=None,
    ) -> np.ndarray:
        """Read only the requested crop when the TIFF backend supports it.

        These images are tiled, so a chunked read touches only the few 256x256
        tiles covering the crop rather than decompressing the whole plane --
        measured at about 100 ms against 13 s for one 3.9 GB channel. *store*
        is a cached zarr view of the page; without one a view is opened here,
        which is correct but pays the page-opening cost every call.
        """
        if store is None and zarr is not None:
            try:
                store = zarr.open(page.aszarr(), mode='r')
            except Exception as e:
                logger.debug(f"Falling back to full page read for {page}: {e}")
        if store is not None:
            try:
                return np.asarray(store[y0:y1, x0:x1])
            except Exception as e:
                logger.debug(f"Chunked read failed for {page}: {e}")

        return page.asarray()[y0:y1, x0:x1]

    def prewarm_batch(self, batch_id: str, requested_channels: list[str] | None = None) -> None:
        """Warm metadata, segmentation index, and TIFF crop access for a batch."""
        if batch_id not in self.file_map:
            return

        paths = self.file_map[batch_id]
        self._load_geojson_index(paths['geojson'])

        try:
            tif = self._get_tiff(batch_id, paths['roi'])
            metadata = self._get_cached_metadata(batch_id, tif)
            page_indices = list(range(len(tif.pages)))
            if requested_channels:
                resolved_indices = self._resolve_channel_indices(requested_channels, metadata['names'])
                if resolved_indices:
                    page_indices = resolved_indices

            if page_indices:
                page = tif.pages[page_indices[0]]
                self._read_page_crop(
                    page, 0, min(1, page.shape[-2]), 0, min(1, page.shape[-1]),
                    store=self._get_page_store(batch_id, tif, page_indices[0]),
                )
        except Exception as e:
            logger.debug(f"Prewarm failed for batch {batch_id}: {e}")

    def _parse_ome_colors(self, ome_xml: str) -> list:
        """Extract channel colors from OME-XML metadata."""
        colors = []
        try:
            # OME-XML usually has a namespace
            # We strip namespaces for simpler parsing or handle them
            # Simple approach: Ignore namespace by filtering tags
            
            root = ET.fromstring(ome_xml)
            
            # Find Pixels element (recursively or direct)
            # Namespace handling: {http://www.openmicroscopy.org/Schemas/OME/2016-06}
            # We'll just search for 'Channel' tags anywhere
            
            for elem in root.iter():
                if 'Channel' in elem.tag and 'Color' in elem.attrib:
                     c_str = elem.attrib['Color']
                     try:
                         # Parse Signed 32-bit Int
                         val = int(c_str)
                         # Convert to unsigned 32-bit
                         val = val & 0xFFFFFFFF
                         
                         # Parse ARGB (Alpha is highest byte)
                         # We want RGB 0-1 triples
                         # (A, R, G, B)
                         a = (val >> 24) & 0xFF
                         r = (val >> 16) & 0xFF
                         g = (val >> 8) & 0xFF
                         b = val & 0xFF
                         
                         colors.append((r/255.0, g/255.0, b/255.0))
                     except:
                         colors.append(None)
                         
            return colors
        except Exception as e:
            logger.error(f"Failed to parse OME colors: {e}")
            return []

    def _parse_ome_channel_names(self, ome_xml: str) -> list:
        """Extract channel Name attributes from OME-XML metadata."""
        names = []
        try:
            root = ET.fromstring(ome_xml)
            # Similar crawl strategy as colors
            for elem in root.iter():
                # Look for Channel elements
                if 'Channel' in elem.tag:
                    # 'Name' attribute is standard in OME
                    name = elem.attrib.get('Name')
                    # Also look for child 'Name' element if attrib missing? Standard is attrib.
                    # Fallback to ID if Name missing? No, let's keep None to indicate missing.
                    if name:
                        names.append(name)
                    else:
                        names.append(f"Channel_{len(names)}") # Fallback index
            return names
        except Exception as e:
            logger.error(f"Failed to parse OME channel names: {e}")
            return []

    def get_cell_context(
        self,
        batch_id: str,
        x: float,
        y: float,
        tile_size: int = 40,
        requested_channels: list[str] | None = None,
    ):
        """
        Retrieves the cell and its context (ROI image crop and neighbor polygons).
        
        Args:
            batch_id: The batch identifier.
            x: Centroid X coordinate.
            y: Centroid Y coordinate.
            tile_size: Radius around the center to extract (default 40px).
        
        Returns:
            dict with:
                'image': numpy array of the crop (C, H, W)
                'target_poly': shapely Polygon of the target cell (relative coordinates)
                'neighbor_polys': list of shapely Polygons of neighbors (relative coordinates)
                'channels': list of channel names (if available in metadata) or None
        """
        if batch_id not in self.file_map:
            print(f"[CellVisualizer] Batch {batch_id} not found in matched files.")
            raise FileNotFoundError(f"No matched files for batch {batch_id}")

        paths = self.file_map[batch_id]
        roi_path = paths['roi']
        seg_path = paths['geojson']

        # 1. Define ROI Bounding Box
        x_min, x_max = int(x - tile_size), int(x + tile_size)
        y_min, y_max = int(y - tile_size), int(y + tile_size)
        
        # 2. Extract Image Crop using tifffile
        # We assume the standard OME-TIFF format where pages represent channels
        # If it's a pyramidal OME-TIFF, we might need a specific series.
        # For simplicity, we assume standard separate-channel-pages or CZYX composite.
        try:
            # Reuses one open handle per batch instead of reopening this
            # multi-gigabyte image for every cell.
            tif = self._get_tiff(batch_id, roi_path)
            metadata = self._get_cached_metadata(batch_id, tif)
            all_channel_colors = metadata['colors']
            all_channel_names = metadata['names']

            page_indices = list(range(len(tif.pages)))
            if requested_channels:
                resolved_indices = self._resolve_channel_indices(requested_channels, all_channel_names)
                if resolved_indices:
                    page_indices = resolved_indices

            image_stack = []
            channel_names = []
            channel_colors = []

            for page_idx in page_indices:
                page = tif.pages[page_idx]
                h, w = page.shape[-2], page.shape[-1]
                y0, y1 = max(0, y_min), min(h, y_max)
                x0, x1 = max(0, x_min), min(w, x_max)

                crop = self._read_page_crop(
                    page, y0, y1, x0, x1, store=self._get_page_store(batch_id, tif, page_idx)
                )
                image_stack.append(crop)
                channel_names.append(all_channel_names[page_idx])
                channel_colors.append(all_channel_colors[page_idx])

            img_data = np.asarray(image_stack)
            
        except Exception as e:
            logger.error(f"Error reading TIFF {roi_path}: {e}")
            raise

        # 3. Extract Polygons
        seg_index = self._load_geojson_index(seg_path)
        all_polys = seg_index['polygons']
        tree = seg_index['tree']
        
        # Create a search box for efficient filtering
        search_box = box(x_min, y_min, x_max, y_max)
        
        target_poly = None
        neighbors = []
        
        # Find target cell (contains centroid) and neighbors (intersect box)
        centroid_pt = Point(x, y)
        candidate_polys = all_polys
        if tree is not None:
            candidate_indices = tree.query(search_box)
            candidate_polys = [all_polys[idx] for idx in candidate_indices]

        for poly in candidate_polys:
            if poly.contains(centroid_pt):
                target_poly = poly
            if poly.intersects(search_box) and (target_poly is None or poly is not target_poly):
                neighbors.append(poly)
                
        # If target not found by strict containment (rare float errors), standard closest
        if target_poly is None and candidate_polys:
            target_poly = min(candidate_polys, key=lambda p: p.distance(centroid_pt))

        # Translate polygons to relative crop coordinates
        def to_relative(p):
            return translate(p, -x_min, -y_min)

        rel_target = to_relative(target_poly) if target_poly else None
        rel_neighbors = [to_relative(p) for p in neighbors if target_poly is None or p is not target_poly]

        return {
            'image': img_data,
            'target_poly': rel_target,
            'neighbor_polys': rel_neighbors,
            'crop_box': (x_min, y_min, x_max, y_max),
            'channel_colors': channel_colors,
            'channel_names': channel_names,
        }

    def generate_visualization(self, 
                             cell_data: dict, 
                             marker_indices: list[int], 
                             marker_names: list[str], 
                             display_size: int = 40,
                             show_neighbors: bool = True):
        """
        Generates a matplotlib Figure with context + top markers using distinct colors.
        
        Args:
            cell_data: Output from get_cell_context
            marker_indices: List of channel indices to plot
            marker_names: Names corresponding to indices
            display_size: Crop radius/size
            show_neighbors: Whether to draw neighbor polygons
        """
        img = cell_data['image']
        target_poly = cell_data['target_poly']
        neighbors = cell_data['neighbor_polys']
        
        num_markers = len(marker_indices)
        cols = num_markers + 1
        
        # Define distinct colors for up to 6 markers (R, G, B, Cyan, Magenta, Yellow)
        # Fallback colors
        fallback_colors = [
            (1, 0, 0),   # Red
            (0, 1, 0),   # Green
            (0, 0, 1),   # Blue
            (0, 1, 1),   # Cyan
            (1, 0, 1),   # Magenta
            (1, 1, 0)    # Yellow
        ]
        
        # Determine colors for SELECTED markers
        selected_colors = []
        used_colors = set()
        
        # Helper to check brightness (simple luma)
        def is_dark(c):
            # c is (r, g, b)
            return (0.299*c[0] + 0.587*c[1] + 0.114*c[2]) < 0.2

        # Check if we have authentic OME colors
        ome_colors = cell_data.get('channel_colors', [])
        
        fallback_idx = 0
        
        for i, idx in enumerate(marker_indices):
            col = None
            # Try OME color first
            if idx < len(ome_colors) and ome_colors[idx] is not None:
                candidate = ome_colors[idx]
                # Check constraints: Not Dark AND Not Duplicate (in this set)
                if not is_dark(candidate) and candidate not in used_colors:
                    col = candidate
            
            # Fallback if invalid or missing
            if col is None:
                # Find next unused fallback
                attempts = 0
                while attempts < len(fallback_colors):
                    fc = fallback_colors[fallback_idx % len(fallback_colors)]
                    fallback_idx += 1
                    attempts += 1
                    if fc not in used_colors:
                        col = fc
                        break
                
                # If still None (all fallbacks used?!), just recycle
                if col is None:
                     col = fallback_colors[i % len(fallback_colors)]
            
            used_colors.add(col)
            selected_colors.append(col)
        
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3), dpi=100)
        if cols == 1: axes = [axes]
        
        # Prepare Overlay (Panel 0)
        # We blend channels by mapping each to its assigned color and adding them
        pad_h, pad_w = img.shape[-2], img.shape[-1]
        overlay_rgb = np.zeros((pad_h, pad_w, 3), dtype=float)
        
        for i, idx in enumerate(marker_indices):
            if idx < img.shape[0]:
                chan = img[idx].astype(float)
                # Normalize 0-1 for blending
                p2, p98 = np.percentile(chan, (2, 98))
                if p98 > p2:
                    chan = (chan - p2) / (p98 - p2)
                else:
                    chan = chan * 0 # Flat signal
                chan = np.clip(chan, 0, 1)
                
                # Add to total RGB
                color = selected_colors[i]
                for c in range(3):
                    overlay_rgb[:, :, c] += chan * color[c]
        
        # Clip final overlay
        overlay_rgb = np.clip(overlay_rgb, 0, 1)
        
        # Define image bounds for strict clipping
        from shapely.geometry import box
        img_box = box(0, 0, pad_w, pad_h)
        
        axes[0].imshow(overlay_rgb)
        axes[0].set_title("Overlay + Seg", fontsize=9, fontweight='bold', color='black')
        axes[0].set_xlim(0, pad_w)
        axes[0].set_ylim(pad_h, 0)
        
        # Draw Polygons on Overlay
        # Target: Bold White
        if target_poly:
            # Clip polygon to image bounds
            clipped_target = target_poly.intersection(img_box)
            if not clipped_target.is_empty:
                if clipped_target.geom_type == 'MultiPolygon':
                    for poly in clipped_target.geoms:
                         x, y = poly.exterior.xy
                         axes[0].plot(x, y, color='white', linewidth=2.5)
                elif clipped_target.geom_type == 'Polygon':
                    x, y = clipped_target.exterior.xy
                    axes[0].plot(x, y, color='white', linewidth=2.5) # Bolder
            
        if show_neighbors:
            for nb in neighbors:
                clipped_nb = nb.intersection(img_box)
                if not clipped_nb.is_empty:
                    if clipped_nb.geom_type == 'MultiPolygon':
                         for poly in clipped_nb.geoms:
                             x, y = poly.exterior.xy
                             axes[0].plot(x, y, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
                    elif clipped_nb.geom_type == 'Polygon':
                        x, y = clipped_nb.exterior.xy
                        axes[0].plot(x, y, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
            
        axes[0].axis('off')

        # Panels 2+: Individual Markers
        for i, (idx, name) in enumerate(zip(marker_indices, marker_names)):
            ax = axes[i + 1]
            color = selected_colors[i]
            
            if idx < img.shape[0]:
                chan_img = img[idx]
                
                # Auto-contrast for display
                vmin, vmax = np.percentile(chan_img, (1, 99))
                
                # Create a specific colormap for this channel (black -> color)
                from matplotlib.colors import LinearSegmentedColormap
                cmap_name = f'custom_{name}'
                # Ensure color is a tuple
                if isinstance(color, str):
                   # Should not happen with our parser but safety check
                   pass 
                
                cm = LinearSegmentedColormap.from_list(cmap_name, ['black', color], N=256)
                
                ax.imshow(chan_img, cmap=cm, vmin=vmin, vmax=vmax)
                ax.set_xlim(0, pad_w)
                ax.set_ylim(pad_h, 0)
                
                # Colored title matching the channel? User asked for black titles.
                ax.set_title(name, fontsize=9, color='black', fontweight='bold')
                
                # Draw subtle target outline -> User requested BOLD segmentation here too
                if target_poly:
                    clipped_target = target_poly.intersection(img_box)
                    if not clipped_target.is_empty:
                        # Use white or matching color? User asked for bold frame. 
                        # Let's use white for visibility against colored background, or the channel color but brighter?
                        # White is safest for visibility.
                        if clipped_target.geom_type == 'MultiPolygon':
                             for poly in clipped_target.geoms:
                                 x, y = poly.exterior.xy
                                 ax.plot(x, y, color='white', linewidth=1.5, alpha=0.9)
                        elif clipped_target.geom_type == 'Polygon':
                             x, y = clipped_target.exterior.xy
                             ax.plot(x, y, color='white', linewidth=1.5, alpha=0.9)

            else:
                ax.text(0.5, 0.5, "Missing", ha='center', va='center', color='red')
            
            ax.axis('off')
            
        plt.tight_layout()
        return fig
