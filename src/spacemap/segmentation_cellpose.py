"""
Segmentation utilities for SPACEMAP pipeline.

This module provides functionality for cell segmentation using various methods,
starting with Cellpose. The structure is designed to be easily extensible 
for additional segmentation methods like DeepCell.
"""

import os
import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET
import cv2
import json
import pandas as pd
import gc
import tqdm
import sys
from datetime import datetime
from typing import Sequence
from skimage.measure import regionprops_table
import rasterio.features
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.affinity import translate as shapely_translate
from shapely.ops import unary_union
import tempfile
import shutil
import zarr
import skimage.feature
if not hasattr(skimage.feature, 'plot_matches'):
    def wrapper_plot_matches(*args, **kwargs):
        names = ['ax', 'image1', 'image2', 'keypoints0', 'keypoints1', 'matches', 'keypoints_color', 'matches_color', 'only_matches', 'alignment']
        bound = {}
        for i, val in enumerate(args):
            if i < len(names):
                bound[names[i]] = val
        bound.update(kwargs)
        ax = bound.get('ax')
        image1 = bound.get('image1')
        image2 = bound.get('image2')
        keypoints0 = bound.get('keypoints0')
        keypoints1 = bound.get('keypoints1')
        matches = bound.get('matches')
        other_kwargs = {}
        for k in ['keypoints_color', 'matches_color', 'only_matches', 'alignment']:
            if k in bound:
                other_kwargs[k] = bound[k]
        return skimage.feature.plot_matched_features(
            image1, image2, keypoints0=keypoints0, keypoints1=keypoints1,
            matches=matches, ax=ax, **other_kwargs
        )
    skimage.feature.plot_matches = wrapper_plot_matches

# Patch skimage.transform geometric transformations to fix 2D translation compatibility with scikit-image 0.25+
try:
    import skimage.transform
    import numpy as np
    import inspect
    def patch_transform_class(cls):
        orig_init = cls.__init__
        def patched_init(self, *args, **kwargs):
            sig = inspect.signature(orig_init)
            bound = sig.bind_partial(self, *args, **kwargs)
            if 'translation' in bound.arguments and bound.arguments['translation'] is not None:
                t = np.asarray(bound.arguments['translation'])
                if t.ndim == 2 and t.shape[0] == 1:
                    bound.arguments['translation'] = t[0]
                elif t.ndim > 1:
                    bound.arguments['translation'] = np.squeeze(t)
            return orig_init(*bound.args, **bound.kwargs)
        cls.__init__ = patched_init

    for cls in [skimage.transform.AffineTransform, skimage.transform.EuclideanTransform, skimage.transform.SimilarityTransform]:
        patch_transform_class(cls)
except Exception:
    pass


from spacemap.background_subtraction_utils import get_channel_info as get_channel_info_any
from spacemap.utils import parse_xml

# Import Cellpose
try:
    from cellpose import models, io, dynamics
    import torch
    CELLPOSE_AVAILABLE = True
    
    # Check GPU availability
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        GPU_AVAILABLE = False
        print("GPU not available, using CPU")
        
except ImportError:
    CELLPOSE_AVAILABLE = False
    GPU_AVAILABLE = False
    print("Warning: Cellpose not available. Install with: pip install cellpose")


# Classifications that are always cut out of a segmentation mask.
DEFAULT_MASK_EXCLUDE_CLASSES = ("artefact", "artifact", "blood_clot", "exclude", "necrosis")


class SegmentationConfig:
    """Configuration class for segmentation parameters."""
    
    def __init__(self):
        # Cellpose specific parameters
        self.diameter = 0.0  # Set to 0 for automatic diameter detection
        self.flow_threshold = 0.8
        self.channels = [1, 3]  # [channel to segment, optional nuclear channel]
        
        # GPU configuration
        self.use_gpu = GPU_AVAILABLE  # Automatically use GPU if available
        self.gpu_device = 0  # GPU device index (for multi-GPU systems)
        
        # General segmentation parameters
        self.tile_size = 2048  # Outer ROI tile size for masked segmentation
        self.tile_overlap = 0.1  # Overlap fraction (0.0 to 1.0)
        # Cellpose tiles within each outer ROI tile. Keeping this modest avoids
        # a single U-Net inference allocating memory for a full 2048 px tile.
        self.cellpose_internal_tile_size = 224
        # Benchmarked on an A100-40GB with a 2048px tile at 83% tissue: batch 8
        # with the flow check on the GPU is 2.2x the old batch-1/CPU-flow default,
        # for the same cells (0.06% spread). Batch 16 was no better.
        self.cellpose_batch_size = 8
        self.adaptive_gpu_memory = True
        self.min_cellpose_internal_tile_size = 128
        # cuDNN was disabled unconditionally to dodge a libcudnn_graph.so
        # mismatch. On a healthy CUDA install that costs a large amount of
        # convolution throughput, so it is now a choice: "auto" probes cuDNN once
        # and falls back only if it actually fails.
        self.cudnn_mode = "auto"          # "auto" | "enabled" | "disabled"
        # Where the flow-error check runs. "auto" uses the GPU when available.
        self.flow_threshold_device = "auto"   # "auto" | "gpu" | "cpu"
        self.min_tissue_fraction = 0.002
        self.tissue_crop_padding = 64
        self.n_pyramid_levels = 3
        self.save_geojson = True
        self.save_npy = False
        self.save_png = False
        
        # Model paths
        self.cellpose_model_path = './data/seg_model/CP_20241014_094617'
        self.cellpose_model_type = 'custom'  # or 'cyto', 'nuclei', etc.
        
        # Optional masking with input GeoJSON
        self.use_input_geojson = False
        self.input_geojson_path = "./data/json/input_mask.geojson"
        # Which classifications in that GeoJSON define the region to segment.
        # Empty means "every class that is not excluded", which preserves the
        # historical behaviour for hand-drawn single-class masks. Set it to
        # ["tissue"] to segment only inside the reviewed valid-tissue polygon
        # of a multi-class annotation file such as *_zones.geojson.
        self.mask_class_names: list[str] = []
        self.mask_exclude_class_names: list[str] = list(DEFAULT_MASK_EXCLUDE_CLASSES)

        # Optional spatial annotation of the extracted cell table. When an
        # annotation GeoJSON is supplied the tissue-domain columns are added in
        # the same pass as feature extraction, so the CSV that lands on disk is
        # already ready for batch correction.
        self.annotation_geojson_path: str | None = None
        self.spatial_rules: list = []
        self.spatial_tissue_label = "tissue"
        self.spatial_min_cells_per_bed_target = 1
        # Restrict every grouped distance to annotations the cell can reach by a
        # straight line inside its own tissue piece. Off by default so existing
        # projects reproduce exactly until the option is switched on.
        self.spatial_require_clean_path = False
        self.spatial_edge_config: dict | None = None


def _is_cuda_oom_error(error: BaseException) -> bool:
    """Return whether an exception is a CUDA allocation failure."""
    message = str(error).lower()
    return "cuda out of memory" in message or "out of memory" in message and "cuda" in message


def get_cellpose_retry_tile_sizes(
    initial_size: int,
    minimum_size: int = 128,
) -> list[int]:
    """Return decreasing Cellpose block sizes compatible with U-Net downsampling."""
    initial = max(32, int(initial_size))
    initial -= initial % 32
    minimum = min(max(32, int(minimum_size)), initial)
    minimum -= minimum % 32
    sizes: list[int] = []
    current = initial
    while current >= minimum:
        if current not in sizes:
            sizes.append(current)
        if current == minimum:
            break
        current = max(minimum, ((current * 3) // 4) // 32 * 32)
    return sizes


def get_channel_names(ome_tiff_path):
    """Extract channel names from OME-TIFF metadata or QPTIFF metadata.

    This uses the same channel-parsing logic as the UI tabs (Tab1/Tab2), so
    `.qptiff` inputs behave consistently with `.ome.tiff`.
    """
    try:
        channel_names, _channel_colors, _channel_meta, _mpp = get_channel_info_any(ome_tiff_path)
        if channel_names:
            return channel_names
    except Exception as e:
        print(f"Error extracting channel names via get_channel_info: {e}")

    # Fallback: attempt OME-XML, then use generic defaults
    try:
        with tiff.TiffFile(ome_tiff_path) as tif:
            ome_xml = tif.ome_metadata
            if not ome_xml:
                raise ValueError("No OME-XML metadata")
            root = ET.fromstring(ome_xml)
            namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            channels = root.findall('.//ome:Channel', namespaces)
            channel_names = [channel.get('Name') for channel in channels]
            if channel_names and None not in channel_names:
                return channel_names
    except Exception:
        pass

    with tiff.TiffFile(ome_tiff_path) as tif:
        return [f'Channel_{i}' for i in range(len(tif.pages))]


def _get_fullres_shape_and_page_indices(tif: tiff.TiffFile) -> tuple[tuple[int, int], list[int]]:
    """Return (H, W) for the largest 2D plane and indices of full-res 2D pages."""
    max_hw = (0, 0)
    for p in tif.pages:
        if getattr(p, 'ndim', 0) == 2:
            h, w = p.shape
            if h * w > max_hw[0] * max_hw[1]:
                max_hw = (int(h), int(w))

    # Fallback for stacks stored as a single 3D page
    if max_hw == (0, 0):
        p0 = tif.pages[0]
        if getattr(p0, 'ndim', 0) == 3:
            # Heuristic: last 2 dims are Y, X for most cases
            max_hw = (int(p0.shape[-2]), int(p0.shape[-1]))
        else:
            max_hw = (int(p0.shape[0]), int(p0.shape[1]))

    fullres_pages = [
        idx
        for idx, p in enumerate(tif.pages)
        if getattr(p, 'ndim', 0) == 2 and tuple(p.shape) == tuple(max_hw)
    ]
    return max_hw, fullres_pages


def _build_channel_page_map(tif: tiff.TiffFile, fullres_pages: list[int]) -> dict[str, int]:
    """Map biomarker/channel name -> TIFF page index for QPTIFF-like files."""
    name_to_page: dict[str, int] = {}
    for idx in fullres_pages:
        try:
            desc = getattr(tif.pages[idx], 'description', '')
            info = parse_xml(desc)
            name = info.get('biomarker')
            if name and name not in name_to_page:
                name_to_page[str(name)] = idx
        except Exception:
            continue
    return name_to_page


def extract_features(mask, images, channel_names):
    """
    Extract features for each segmented cell.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Segmentation mask with labeled regions
    images : list of numpy.ndarray
        List of channel images
    channel_names : list of str
        Names of the channels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted features for each cell
    """
    # print("Extracting morphological features...")
    # First, get all morphological properties. This is done once.
    morph_props = regionprops_table(
        mask,
        properties=[
            'label', 'area', 'major_axis_length', 'minor_axis_length',
            'eccentricity', 'solidity', 'extent', 'orientation', 'centroid'
        ]
    )
    morph_df = pd.DataFrame(morph_props)
    morph_df.rename(columns={'centroid-0': 'y_centroid', 'centroid-1': 'x_centroid'}, inplace=True)

    # Mean intensity per label, one vectorized pass per channel.
    #
    # regionprops_table(intensity_image=...) walks the whole mask separately for
    # every channel, so a 41-plex tile paid ~41 full traversals - about as much
    # time as the segmentation itself. Summing with bincount over the flattened
    # label map gives exactly the same means (verified bit-identical) roughly
    # 6.5x faster, because the per-label pixel counts are computed only once.
    combined_df = morph_df.set_index('label')
    if images is not None and len(images) > 0:
        flat_labels = np.asarray(mask).reshape(-1)
        pixel_counts = np.bincount(flat_labels)
        present_labels = np.nonzero(pixel_counts)[0]
        present_labels = present_labels[present_labels > 0]

        if present_labels.size:
            divisor = pixel_counts[present_labels].astype(np.float64)
            intensity_columns = {}
            for image, channel_name in zip(images, channel_names):
                sums = np.bincount(
                    flat_labels,
                    weights=np.asarray(image).reshape(-1).astype(np.float64, copy=False),
                    minlength=pixel_counts.size,
                )
                intensity_columns[channel_name] = sums[present_labels] / divisor
            intensity_df = pd.DataFrame(intensity_columns, index=pd.Index(present_labels, name='label'))
            combined_df = combined_df.join(intensity_df)

    return combined_df.reset_index()


def generate_geojson_from_mask(mask, output_path, *, offset_y: int = 0, offset_x: int = 0, dissolve_by_label: bool = False):
    """
    Generate GeoJSON file from segmentation mask.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Segmentation mask with labeled regions
    output_path : str
        Path to save the GeoJSON file
        
    Returns:
    --------
    str
        Path to the saved GeoJSON file
    """
    # No "skip if the file already exists" guard here. Reaching this function
    # means the caller has just produced fresh masks, so anything already at
    # this path is a stale export from an earlier run. Skipping the write left
    # the cell boundaries out of sync with the features CSV every time a
    # segmentation was re-run with overwrite enabled. Whether an image is
    # re-segmented at all is decided by segment_image_cellpose(), which returns
    # early on `not overwrite and check_segmentation_outputs_exist(...)`.

    print("Exporting cell boundaries as GeoJSON...")
    t0 = datetime.now()
    results = rasterio.features.shapes(
        mask.astype(np.int32), 
        mask=(mask > 0), 
        connectivity=8
    )

    def _round_int_coords(geom_dict: dict) -> dict:
        gtype = geom_dict.get('type')
        coords = geom_dict.get('coordinates')
        if gtype == 'Polygon':
            geom_dict['coordinates'] = [
                [[int(round(x)), int(round(y))] for (x, y) in ring]
                for ring in coords
            ]
        elif gtype == 'MultiPolygon':
            geom_dict['coordinates'] = [
                [
                    [[int(round(x)), int(round(y))] for (x, y) in ring]
                    for ring in poly
                ]
                for poly in coords
            ]
        return geom_dict

    if dissolve_by_label:
        geoms_by_label: dict[int, list] = {}
        for geom, value in results:
            if value == 0:
                continue
            lbl = int(value)
            g = shape(geom)
            if offset_x or offset_y:
                g = shapely_translate(g, xoff=offset_x, yoff=offset_y)
            geoms_by_label.setdefault(lbl, []).append(g)

        features = []
        for lbl, parts in geoms_by_label.items():
            merged = unary_union(parts) if len(parts) > 1 else parts[0]
            geom_dict = mapping(merged)
            geom_dict = _round_int_coords(geom_dict)
            features.append({
                "type": "Feature",
                "properties": {
                    "label": int(lbl),
                    "area": float(merged.area),
                },
                "geometry": geom_dict,
            })
    else:
        features = []
        for geom, value in results:
            if value == 0:
                continue

            poly = shape(geom)
            if offset_x or offset_y:
                poly = shapely_translate(poly, xoff=offset_x, yoff=offset_y)
            geom_dict = mapping(poly)
            geom_dict = _round_int_coords(geom_dict)

            feature = {
                "type": "Feature",
                "properties": {
                    "label": int(value),
                    "area": float(poly.area),
                },
                "geometry": geom_dict,
            }
            features.append(feature)

    geojson_dict = {"type": "FeatureCollection", "features": features}

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as a single line with no extra spaces
    with open(output_path, 'w') as f:
        json.dump(geojson_dict, f, separators=(',', ':'))
    
    print(f"GeoJSON export time: {datetime.now() - t0}")
    print(f"GeoJSON saved to: {output_path}")
    return output_path


def create_tissue_mask(dapi_image, geojson_path):
    """
    Create a tissue mask from GeoJSON annotations.
    
    Parameters:
    -----------
    dapi_image : numpy.ndarray
        DAPI image to get dimensions
    geojson_path : str
        Path to GeoJSON file with tissue annotations
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where 1 indicates tissue regions
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    tissue_polygons = []
    exclude_polygons = []
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        annotation_name = 'tissue' # Default to tissue/ROI
        
        # Try to determine class from various properties
        if 'classification' in props and props['classification'] and 'name' in props['classification']:
            annotation_name = props['classification']['name'].lower()
        elif 'label' in props:
            annotation_name = props['label'].lower()
        elif 'name' in props:
            annotation_name = props['name'].lower()
            
        geom = shape(feature['geometry'])
        
        if annotation_name in ['artefact', 'blood_clot', 'exclude', 'necrosis']:
            exclude_polygons.append(geom)
        else:
            # Treat everything else as a positive ROI to segment
            tissue_polygons.append(geom)
    
    # Create union of exclusion regions
    exclude_union = unary_union(exclude_polygons) if exclude_polygons else None
    
    # Process tissue polygons by removing excluded regions
    processed_polygons = []
    for tissue_poly in tissue_polygons:
        if not tissue_poly.is_empty:
            if exclude_union:
                processed_poly = tissue_poly.difference(exclude_union)
            else:
                processed_poly = tissue_poly
            processed_polygons.append(processed_poly)
    
    # Create mask image
    mask_image = np.zeros(dapi_image.shape, dtype=np.uint8)
    
    for poly in processed_polygons:
        for subpoly in iter_polygons(poly):
            exterior_coords = np.array(subpoly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask_image, [exterior_coords], 1)

            # Handle holes
            for interior in subpoly.interiors:
                interior_coords = np.array(interior.coords, dtype=np.int32)
                cv2.fillPoly(mask_image, [interior_coords], 0)
    
    return mask_image


def get_tissue_mask_for_tile(tile_shape, tile_offset, geojson_path):
    """
    Generate tissue mask for a specific tile.
    
    Parameters:
    -----------
    tile_shape : tuple
        (height, width) of the tile
    tile_offset : tuple
        (y, x) offset of the tile in the full image
    geojson_path : str
        Path to GeoJSON file
        
    Returns:
    --------
    numpy.ndarray
        Binary mask for the tile
    """
    if not os.path.exists(geojson_path):
        return np.ones(tile_shape, dtype=np.uint8)
        
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    tissue_polygons = []
    exclude_polygons = []
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        annotation_name = 'tissue' # Default to tissue/ROI
        
        # Try to determine class from various properties
        if 'classification' in props and props['classification'] and 'name' in props['classification']:
            annotation_name = props['classification']['name'].lower()
        elif 'label' in props:
            annotation_name = props['label'].lower()
        elif 'name' in props:
            annotation_name = props['name'].lower()
            
        geom = shape(feature['geometry'])
        
        if annotation_name in ['artefact', 'blood_clot', 'exclude', 'necrosis']:
            exclude_polygons.append(geom)
        else:
            # Treat everything else as a positive ROI to segment
            tissue_polygons.append(geom)
    
    # Create union of exclusion regions
    exclude_union = unary_union(exclude_polygons) if exclude_polygons else None
    
    # Process tissue polygons
    processed_polygons = []
    for tissue_poly in tissue_polygons:
        if not tissue_poly.is_empty:
            if exclude_union:
                processed_poly = tissue_poly.difference(exclude_union)
            else:
                processed_poly = tissue_poly
            processed_polygons.append(processed_poly)
            
    # Create mask image for the tile
    mask_image = np.zeros(tile_shape, dtype=np.uint8)
    tile_y, tile_x = tile_offset
    tile_h, tile_w = tile_shape
    
    # Define tile polygon in global coordinates
    tile_box = Polygon([
        (tile_x, tile_y),
        (tile_x + tile_w, tile_y),
        (tile_x + tile_w, tile_y + tile_h),
        (tile_x, tile_y + tile_h)
    ])
    
    for poly in processed_polygons:
        # Check if polygon intersects with tile
        if poly.intersects(tile_box):
            # Clip polygon to tile
            intersection = poly.intersection(tile_box)
            
            # Translate to tile coordinates
            def translate_coords(x, y, z=None):
                return (x - tile_x, y - tile_y)
            
            from shapely.ops import transform
            local_poly = transform(translate_coords, intersection)
            
            if isinstance(local_poly, (Polygon, MultiPolygon)):
                polys_to_fill = [local_poly] if isinstance(local_poly, Polygon) else list(local_poly.geoms)
                
                for subpoly in polys_to_fill:
                    exterior_coords = np.array(subpoly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(mask_image, [exterior_coords], 1)
                    
                    for interior in subpoly.interiors:
                        interior_coords = np.array(interior.coords, dtype=np.int32)
                        cv2.fillPoly(mask_image, [interior_coords], 0)
    
    return mask_image


# Suffixes an annotation file may carry, most specific first. `*_zones` is the
# final output of the Annotation tab and therefore the preferred match.
ANNOTATION_GEOJSON_SUFFIXES = (
    "_zones",
    "_macro_domains",
    "_merged",
    "_valid_tissue_from_clusters_highres",
    "_valid_tissue",
    "",
)


def image_stem(image_path: str) -> str:
    """Strip image extensions so `P11_roi_1.ome.tiff` becomes `P11_roi_1`."""
    name = os.path.basename(str(image_path))
    lowered = name.lower()
    for extension in (".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif"):
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return os.path.splitext(name)[0]


def find_annotation_geojson(
    image_path: str,
    geojson_dir: str,
    suffixes: Sequence[str] = ANNOTATION_GEOJSON_SUFFIXES,
) -> str | None:
    """Match an ROI image to its annotation GeoJSON by file name.

    ``P11_roi_1.ome.tiff`` matches ``P11_roi_1_zones.geojson`` first, then the
    other known annotation outputs, then any file whose stem starts with the
    image stem. Returns None when nothing matches.
    """
    if not geojson_dir or not os.path.isdir(geojson_dir):
        return None
    stem = image_stem(image_path)

    for suffix in suffixes:
        candidate = os.path.join(geojson_dir, f"{stem}{suffix}.geojson")
        if os.path.isfile(candidate):
            return candidate

    # Fall back to a prefix match so a custom suffix still resolves.
    prefixed = sorted(
        name for name in os.listdir(geojson_dir)
        if name.lower().endswith(".geojson") and name.startswith(stem)
    )
    if prefixed:
        return os.path.join(geojson_dir, prefixed[0])
    return None


def get_geojson_class_names(geojson_path: str) -> list[str]:
    """Return the classification names available in an annotation GeoJSON."""
    try:
        with open(geojson_path, "r") as geojson_file:
            geojson_data = json.load(geojson_file)
    except Exception:
        return []
    names = set()
    for feature in geojson_data.get("features", []):
        name = _feature_classification(feature)
        if name:
            names.add(name)
    return sorted(names)


def _feature_classification(feature: dict) -> str | None:
    """Classification name of one GeoJSON feature, or None when unclassified."""
    properties = feature.get("properties", {}) or {}
    classification = properties.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    if isinstance(classification, str) and classification:
        return classification
    for key in ("label", "name"):
        if properties.get(key):
            return str(properties[key])
    return None


def load_tissue_geometry(
    geojson_path: str,
    include_classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None,
):
    """Load the geometry that defines where segmentation may run.

    ``include_classes`` restricts the mask to specific classifications, which is
    what makes a multi-class annotation file usable: pointing it at ``tissue``
    segments only inside the reviewed valid-tissue polygon of a ``*_zones.geojson``
    while ignoring the Epithelia/Stroma/zone layers in the same file. When it is
    empty every non-excluded class contributes, matching the historical
    behaviour for single-class hand-drawn masks.
    """
    with open(geojson_path, "r") as geojson_file:
        geojson_data = json.load(geojson_file)

    include = {str(name).lower() for name in (include_classes or []) if str(name).strip()}
    exclude = {
        str(name).lower()
        for name in (DEFAULT_MASK_EXCLUDE_CLASSES if exclude_classes is None else exclude_classes)
    }

    tissue_polygons = []
    exclusion_polygons = []
    for feature in geojson_data.get("features", []):
        annotation_name = (_feature_classification(feature) or "tissue").lower()
        geometry_dict = feature.get("geometry")
        if not geometry_dict:
            continue
        geometry = shape(geometry_dict)
        if geometry.is_empty:
            continue
        if annotation_name in exclude:
            exclusion_polygons.append(geometry)
        elif not include or annotation_name in include:
            tissue_polygons.append(geometry)

    if not tissue_polygons:
        available = ", ".join(get_geojson_class_names(geojson_path)) or "none"
        if include:
            raise ValueError(
                f"No polygons of class {sorted(include)} in {os.path.basename(geojson_path)}. "
                f"Classes present: {available}."
            )
        raise ValueError("The tissue GeoJSON contains no inclusion polygons.")
    tissue_geometry = unary_union(tissue_polygons)
    if exclusion_polygons:
        tissue_geometry = tissue_geometry.difference(unary_union(exclusion_polygons))
    if tissue_geometry.is_empty:
        raise ValueError("The tissue GeoJSON is empty after applying exclusion polygons.")
    return tissue_geometry


def iter_polygons(geometry):
    """Yield every Polygon inside any geometry, recursing into collections.

    Clipping a tissue geometry to a tile very often returns a
    ``GeometryCollection`` (polygons plus the touching edges and points).
    Handling only Polygon/MultiPolygon silently drops those tiles.
    """
    if geometry is None or geometry.is_empty:
        return
    geometry_type = geometry.geom_type
    if geometry_type == "Polygon":
        yield geometry
    elif geometry_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from iter_polygons(part)
    # Lines and points carry no area and contribute nothing to a mask.


def get_tissue_mask_for_tile_geometry(tile_shape, tile_offset, tissue_geometry):
    """Rasterize an already-loaded tissue geometry into one outer tile."""
    tile_height, tile_width = tile_shape
    tile_y, tile_x = tile_offset
    mask_image = np.zeros((tile_height, tile_width), dtype=np.uint8)
    tile_box = Polygon([
        (tile_x, tile_y),
        (tile_x + tile_width, tile_y),
        (tile_x + tile_width, tile_y + tile_height),
        (tile_x, tile_y + tile_height),
    ])
    if not tissue_geometry.intersects(tile_box):
        return mask_image

    intersection = tissue_geometry.intersection(tile_box)
    if intersection.is_empty:
        return mask_image

    from shapely.ops import transform

    def translate_to_tile(x, y, z=None):
        return x - tile_x, y - tile_y

    local_geometry = transform(translate_to_tile, intersection)
    for polygon in iter_polygons(local_geometry):
        if polygon.is_empty:
            continue
        exterior = np.asarray(polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask_image, [exterior], 1)
        for interior in polygon.interiors:
            cv2.fillPoly(mask_image, [np.asarray(interior.coords, dtype=np.int32)], 0)
    return mask_image


def get_mask_crop_bounds(tissue_mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    """Return the smallest padded tile-local rectangle covering tissue pixels."""
    tissue_rows, tissue_columns = np.where(tissue_mask > 0)
    if len(tissue_rows) == 0:
        return None
    y0 = max(0, int(tissue_rows.min()) - int(padding))
    y1 = min(tissue_mask.shape[0], int(tissue_rows.max()) + 1 + int(padding))
    x0 = max(0, int(tissue_columns.min()) - int(padding))
    x1 = min(tissue_mask.shape[1], int(tissue_columns.max()) + 1 + int(padding))
    return y0, y1, x0, x1


def _open_series_zarr_array(series, level: int = 0):
    """Return the requested TIFF series pyramid level as a Zarr array."""
    node = zarr.open(series.aszarr(level=level), mode="r")
    if hasattr(node, "shape"):
        return node
    if "0" in node:
        return node["0"]
    array_keys = list(node.array_keys())
    if not array_keys:
        raise ValueError("TIFF Zarr store contains no readable arrays.")
    return node[array_keys[0]]


class CellposeSegmenter:
    """Cellpose-based cell segmentation."""
    
    def __init__(self, config):
        """
        Initialize Cellpose segmenter.
        
        Parameters:
        -----------
        config : SegmentationConfig
            Configuration object with segmentation parameters
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is required but not available")
            
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Cellpose model."""
        print("Loading Cellpose model...")
        
        # Print GPU status
        if self.config.use_gpu and GPU_AVAILABLE:
            print(f"Using GPU: {torch.cuda.get_device_name(self.config.gpu_device)}")
            self._configure_cudnn()
        else:
            print("Using CPU")
        
        if os.path.exists(self.config.cellpose_model_path):
            # Load custom model
            self.model = models.CellposeModel(
                pretrained_model=self.config.cellpose_model_path,
                gpu=self.config.use_gpu and GPU_AVAILABLE,
                device=torch.device(f'cuda:{self.config.gpu_device}' if self.config.use_gpu and GPU_AVAILABLE else 'cpu')
            )
            print(f"Loaded custom model from: {self.config.cellpose_model_path}")
        else:
            # Load pretrained model
            self.model = models.Cellpose(
                model_type=self.config.cellpose_model_type,
                gpu=self.config.use_gpu and GPU_AVAILABLE,
                device=torch.device(f'cuda:{self.config.gpu_device}' if self.config.use_gpu and GPU_AVAILABLE else 'cpu')
            )
            print(f"Loaded pretrained model: {self.config.cellpose_model_type}")
            
        # Disable MKLDNN on CPU to prevent OneDNN "could not create a primitive descriptor for the sum primitive" crash
        if not (self.config.use_gpu and GPU_AVAILABLE):
            try:
                if hasattr(self.model, 'net') and self.model.net is not None:
                    self.model.net.mkldnn = False
                    print("Disabled MKLDNN on CPU for CellposeModel.")
                elif hasattr(self.model, 'cp') and hasattr(self.model.cp, 'net') and self.model.cp.net is not None:
                    self.model.cp.net.mkldnn = False
                    print("Disabled MKLDNN on CPU for Cellpose.")
            except Exception as e:
                print(f"Warning: Could not disable MKLDNN: {e}")
    
    def _configure_cudnn(self) -> None:
        """Enable cuDNN when it works; disable it only if it actually fails.

        Running convolutions without cuDNN falls back to much slower kernels.
        The historical unconditional disable was a workaround for a broken
        libcudnn_graph.so; probing costs milliseconds and keeps the fast path
        when the install is healthy.
        """
        mode = str(getattr(self.config, "cudnn_mode", "auto")).lower()
        if mode == "disabled":
            torch.backends.cudnn.enabled = False
            print("cuDNN disabled by configuration.")
            return
        try:
            torch.backends.cudnn.enabled = True
            if mode == "auto":
                # A tiny convolution is enough to surface a broken cuDNN load.
                device = torch.device(f"cuda:{self.config.gpu_device}")
                probe = torch.zeros((1, 1, 8, 8), device=device)
                weight = torch.zeros((1, 1, 3, 3), device=device)
                torch.nn.functional.conv2d(probe, weight, padding=1)
                torch.cuda.synchronize()
            # benchmark picks the fastest algorithm for the block size, which is
            # constant across tiles here, so the one-off autotune pays for itself.
            torch.backends.cudnn.benchmark = True
            print("cuDNN enabled (benchmark autotuning on).")
        except Exception as error:
            torch.backends.cudnn.enabled = False
            print(f"cuDNN probe failed ({error}); continuing without cuDNN.")

    def segment(self, image_stack, progress_callback=None, **kwargs):
        """
        Perform segmentation on image stack.
        
        Parameters:
        -----------
        image_stack : numpy.ndarray
            Image stack for segmentation (H, W, C)
        progress_callback : callable, optional
            Callback for granular progress updates (0.0 to 1.0)
        **kwargs : dict
            Additional arguments passed to cellpose.eval
            
        Returns:
        --------
        tuple
            (masks, flows, styles) from Cellpose
        """
        # print("Running Cellpose segmentation...")
        h, w = image_stack.shape[:2]
        # print(f"Image shape: {image_stack.shape}")
        
        # Estimate tiles for user feedback
        # Cellpose default tile size is often 224 or based on diameter
        # We assume ~224 for estimation if diameter is 0 (auto)
        est_tile_size = 224
        if self.config.diameter > 0:
            est_tile_size = int(self.config.diameter * 2) # Rough guess
            
        n_tiles_h = np.ceil(h / est_tile_size)
        n_tiles_w = np.ceil(w / est_tile_size)
        est_total_tiles = int(n_tiles_h * n_tiles_w)
        # print(f"Estimated number of tiles: ~{est_total_tiles} (based on size {est_tile_size})")
        
        # Only collect Python garbage here. Calling torch.cuda.empty_cache() on
        # every tile returns the caching allocator's blocks to the driver and
        # forces it to re-acquire them, which is pure overhead; it is still
        # called on the OOM path where it actually helps.
        gc.collect()
        if self.config.use_gpu and GPU_AVAILABLE:
            mem_free, mem_total = torch.cuda.mem_get_info()
            # print(f"GPU Memory before run: {mem_free/1024**3:.2f}GB free / {mem_total/1024**3:.2f}GB total")
        
        # Context manager to patch tqdm for progress reporting
        class TqdmPatcher:
            def __init__(self, callback):
                self.callback = callback
                self.original_tqdm = tqdm.tqdm
                self.patched_modules = []
                
            def __enter__(self):
                if not self.callback:
                    return
                
                cb = self.callback
                class ProxyTqdm(self.original_tqdm):
                    def update(self, n=1):
                        super().update(n)
                        if self.total and self.total > 0:
                            cb(self.n / self.total)
                            
                # Patch global tqdm
                tqdm.tqdm = ProxyTqdm
                
                # Patch cellpose modules if they imported tqdm
                # We try to find where tqdm is used in cellpose
                modules_to_patch = ['cellpose.models', 'cellpose.dynamics', 'cellpose.core']
                for mod_name in modules_to_patch:
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        if hasattr(mod, 'tqdm'):
                            # Check if it's the class we want to patch
                            if mod.tqdm is self.original_tqdm:
                                setattr(mod, 'tqdm', ProxyTqdm)
                                self.patched_modules.append((mod, 'tqdm'))
                            # Or if it's 'trange' which is often just tqdm(range(...))
                            # But usually trange is imported separately.
                            
                return self
                
            def __exit__(self, *args):
                if not self.callback:
                    return
                # Restore global
                tqdm.tqdm = self.original_tqdm
                # Restore modules
                for mod, attr in self.patched_modules:
                    setattr(mod, attr, self.original_tqdm)

        # Run segmentation with flow_threshold=None to avoid GPU OOM during flow error check
        # We will perform the flow error check manually on CPU afterwards
        print("Step 1: Running network prediction (flow_threshold=None)...")
        
        # Determine tiling parameters. `tile` is not present in Cellpose 3's
        # public eval signature, but `bsize` is. A small bsize keeps Cellpose
        # from treating an entire outer ROI tile as one GPU inference block.
        use_tile = kwargs.get('tile', True)
        tile_overlap = kwargs.get('tile_overlap', 0.1)
        batch_size = max(1, int(kwargs.get('batch_size', self.config.cellpose_batch_size)))
        requested_bsize = max(
            32,
            int(kwargs.get('bsize', self.config.cellpose_internal_tile_size)),
        )
        requested_bsize -= requested_bsize % 32
        retry_floor = int(self.config.min_cellpose_internal_tile_size)
        if self.config.use_gpu and GPU_AVAILABLE and self.config.adaptive_gpu_memory:
            if requested_bsize > 512:
                print(
                    f"Adaptive GPU memory safety capped Cellpose internal block "
                    f"from {requested_bsize}px to 512px."
                )
                requested_bsize = 512
            # Scale the batch to the memory actually free rather than forcing 1.
            # Forcing 1 serialises ~85 blocks per 2048px tile, which leaves a
            # 40 GB card almost idle. The OOM retry below remains the safety net.
            try:
                free_bytes, _total = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
            except Exception:
                free_gb = 0.0
            if free_gb > 0:
                per_item_gb = 0.12 * (requested_bsize / 224.0) ** 2
                affordable = max(1, int((free_gb * 0.5) / max(per_item_gb, 1e-6)))
                if batch_size > affordable:
                    print(
                        f"Adaptive GPU memory safety reduced Cellpose GPU batch size "
                        f"from {batch_size} to {affordable} ({free_gb:.1f} GB free)."
                    )
                    batch_size = affordable
                else:
                    print(
                        f"Cellpose GPU batch size {batch_size} at {requested_bsize}px "
                        f"blocks ({free_gb:.1f} GB free)."
                    )
            if retry_floor >= requested_bsize:
                retry_floor = 128
        
        # Build eval arguments dynamically based on signature compatibility
        eval_kwargs = {
            "diameter": self.config.diameter,
            "flow_threshold": None,  # Skip GPU-intensive flow error check
            "channels": self.config.channels,
            "tile_overlap": tile_overlap,
            "batch_size": batch_size
        }
        
        # Check if 'tile' is supported by the underlying CellposeModel
        eval_func = self.model.cp.eval if hasattr(self.model, 'cp') and hasattr(self.model.cp, 'eval') else self.model.eval
        import inspect
        try:
            sig = inspect.signature(eval_func)
            if 'tile' in sig.parameters:
                eval_kwargs['tile'] = use_tile
            if 'bsize' in sig.parameters:
                eval_kwargs['bsize'] = requested_bsize
            elif not use_tile:
                # In Cellpose 3.x, tile is not supported. If tile=False is requested (e.g. for manual tiling),
                # the public API does not expose a way to fully disable tiling.
                # Keep the compatibility fallback memory-safe rather than using
                # one block as large as the whole ROI tile.
                try:
                    cpm = self.model.cp if hasattr(self.model, 'cp') else self.model
                    diam_mean = getattr(cpm, 'diam_mean', 30.0)
                    diam_labels = getattr(cpm, 'diam_labels', 30.0)
                    if diam_mean <= 0: diam_mean = 30.0
                    if diam_labels <= 0: diam_labels = 30.0
                    
                    if self.config.diameter is not None and self.config.diameter > 0:
                        rescale = diam_mean / self.config.diameter
                    else:
                        rescale = diam_mean / diam_labels
                except Exception:
                    rescale = 1.0
                
                eval_kwargs['bsize'] = requested_bsize
        except Exception as e:
            print(f"Warning: Could not inspect Cellpose model signature: {e}. Falling back to default parameters.")

        retry_sizes = [requested_bsize]
        if (
            self.config.use_gpu
            and GPU_AVAILABLE
            and self.config.adaptive_gpu_memory
            and 'bsize' in eval_kwargs
        ):
            retry_sizes = get_cellpose_retry_tile_sizes(
                requested_bsize,
                retry_floor,
            )

        # Recover from OOM by halving the batch first: it frees memory
        # proportionally while keeping the (faster) larger block size.
        attempts: list[tuple[int, int]] = []
        shrinking_batch = batch_size
        while shrinking_batch > 1:
            attempts.append((retry_sizes[0], shrinking_batch))
            shrinking_batch = max(1, shrinking_batch // 2)
        for size in retry_sizes:
            attempts.append((size, 1))

        last_oom: BaseException | None = None
        for attempt_index, (bsize, attempt_batch) in enumerate(attempts):
            if 'bsize' in eval_kwargs:
                eval_kwargs['bsize'] = bsize
            eval_kwargs['batch_size'] = attempt_batch
            try:
                print(
                    f"Cellpose internal block size: {bsize}px "
                    f"(attempt {attempt_index + 1}/{len(attempts)}, batch size {attempt_batch})."
                )
                with TqdmPatcher(progress_callback):
                    evaluation = self.model.eval(
                        image_stack,
                        **eval_kwargs,
                    )
                # `models.CellposeModel` (custom weights) returns
                # (masks, flows, styles); `models.Cellpose` (pretrained
                # model_type) additionally returns the estimated diameters.
                masks, flows, styles = evaluation[0], evaluation[1], evaluation[2]
                break
            except RuntimeError as error:
                if not _is_cuda_oom_error(error) or attempt_index == len(attempts) - 1:
                    raise
                last_oom = error
                next_bsize, next_batch = attempts[attempt_index + 1]
                print(
                    f"CUDA out of memory at block {bsize}px / batch {attempt_batch}. "
                    f"Clearing GPU cache and retrying with {next_bsize}px / batch {next_batch}."
                )
                gc.collect()
                torch.cuda.empty_cache()
        else:
            raise RuntimeError(
                "CUDA out of memory after reducing Cellpose internal tile size. "
                "Use a smaller internal block size or disable GPU."
            ) from last_oom
        
        # Perform flow error check on CPU if needed
        if self.config.flow_threshold is not None and self.config.flow_threshold > 0:
            print(f"Step 2: Applying flow threshold {self.config.flow_threshold}...")
            try:
                # flows output format from Cellpose eval: [plot_flows, cellprob, dP, p]
                # We need dP (flow field) which is usually at index 2
                dP = None
                if isinstance(flows, list):
                    # print(f"Flows list length: {len(flows)}")
                    for i, f in enumerate(flows):
                        if hasattr(f, 'shape'):
                            pass # print(f"Flows[{i}] shape: {f.shape}")
                    
                    # Heuristic to find dP (channels=2 or 3, matching H,W)
                    for f in flows:
                        if hasattr(f, 'shape') and f.ndim == 3 and (f.shape[0] == 2 or f.shape[0] == 3):
                            if f.shape[1] == masks.shape[0] and f.shape[2] == masks.shape[1]:
                                dP = f
                                break
                    
                    if dP is None and len(flows) > 2:
                         dP = flows[2]
                else:
                    # Fallback if structure is different
                    dP = flows[0]
                
                if dP is None:
                    print("Warning: Could not identify flow field (dP) in output.")
                    raise ValueError("dP not found")

                # print(f"Masks shape: {masks.shape}, dP shape: {dP.shape}")

                # Ensure data is on CPU (numpy)
                if hasattr(masks, 'cpu'): masks = masks.cpu().numpy()
                if hasattr(dP, 'cpu'): dP = dP.cpu().numpy()
                if isinstance(masks, torch.Tensor): masks = masks.numpy()
                if isinstance(dP, torch.Tensor): dP = dP.numpy()

                # Fix dimension mismatch if present
                if dP.shape[1:] != masks.shape:
                    # print(f"Resizing flows {dP.shape} to match masks {masks.shape}")
                    # dP is (C, H, W). We need to resize spatial dims.
                    # Transpose to (H, W, C) for cv2
                    dP_trans = np.moveaxis(dP, 0, -1)
                    # cv2.resize expects (width, height)
                    dP_resized = cv2.resize(dP_trans, (masks.shape[1], masks.shape[0]), interpolation=cv2.INTER_LINEAR)
                    # Transpose back to (C, H, W)
                    dP = np.moveaxis(dP_resized, -1, 0)

                # remove_bad_flow_masks runs where its inputs live. On a tile with
                # thousands of masks the CPU pass is a substantial slice of the
                # per-tile time, so use the GPU when one is available.
                flow_device = str(getattr(self.config, "flow_threshold_device", "auto")).lower()
                use_gpu_flow = (
                    flow_device in {"auto", "gpu"}
                    and self.config.use_gpu
                    and GPU_AVAILABLE
                )
                result = None
                if use_gpu_flow:
                    try:
                        device = torch.device(f"cuda:{self.config.gpu_device}")
                        result = dynamics.remove_bad_flow_masks(
                            masks, dP, threshold=self.config.flow_threshold, device=device
                        )
                    except Exception as flow_error:
                        print(f"    GPU flow check unavailable ({flow_error}); using CPU.")
                        result = None
                if result is None:
                    result = dynamics.remove_bad_flow_masks(
                        masks,
                        dP,
                        threshold=self.config.flow_threshold
                    )
                
                if isinstance(result, tuple):
                    if len(result) >= 1:
                        masks = result[0]
                    if len(result) >= 2:
                        errors = result[1]
                else:
                    # Assume it returned just the masks
                    masks = result
                    
                print("Flow threshold applied successfully.")
            except Exception as e:
                print(f"Warning: Failed to apply flow threshold on CPU: {e}")
                print("Returning masks without flow threshold filtering.")

        # Clear memory after segmentation
        if self.config.use_gpu and GPU_AVAILABLE:
            torch.cuda.empty_cache()
        
        print(f"Segmentation completed. Found {len(np.unique(masks)) - 1} cells.")
        return masks, flows, styles


def apply_spatial_annotation(features_df, config, image_name: str = ""):
    """Add tissue-domain columns to a freshly extracted cell table.

    Runs in the same pass as feature extraction so the CSV written to disk is
    already annotated and ready for batch correction, instead of needing a
    separate join later. Returns ``(dataframe, summary_or_None)``.
    """
    geojson_path = getattr(config, "annotation_geojson_path", None)
    if not geojson_path or not os.path.exists(str(geojson_path)):
        return features_df, None
    rules_input = getattr(config, "spatial_rules", None) or []
    edge_input = getattr(config, "spatial_edge_config", None)
    if not rules_input and not edge_input:
        return features_df, None

    try:
        from spacemap.cell_spatial_annotation import (
            EdgeTagConfig,
            SpatialFeatureRule,
            annotate_cells,
            rules_from_records,
        )

        if rules_input and isinstance(rules_input[0], SpatialFeatureRule):
            rules = list(rules_input)
        else:
            rules = rules_from_records(rules_input)

        edge_config = None
        if edge_input:
            edge_config = EdgeTagConfig(
                reference=str(edge_input.get("reference", config.spatial_tissue_label)),
                buffer_px=float(edge_input.get("buffer", edge_input.get("buffer_px", 100.0))),
                column=str(edge_input.get("column", "edge_cell")),
            )

        annotated, summary = annotate_cells(
            features_df,
            str(geojson_path),
            rules,
            tissue_label=str(getattr(config, "spatial_tissue_label", "tissue")),
            edge_config=edge_config,
            min_cells_per_bed_target=max(
                int(getattr(config, "spatial_min_cells_per_bed_target", 1)), 1
            ),
            require_clean_path=bool(getattr(config, "spatial_require_clean_path", False)),
        )
        added = len(summary.get("columns", []))
        print(f"Spatial annotation: added {added} column(s) from {os.path.basename(str(geojson_path))}")
        for warning in summary.get("warnings", []):
            print(f"  Warning: {warning}")
        return annotated, summary
    except Exception as error:
        # Never lose a completed segmentation because the annotation step failed.
        print(f"Warning: spatial annotation failed for {image_name}: {error}")
        return features_df, {"error": str(error)}


def read_preview_plane(ome_tiff_path, channel_index: int, target_shape) -> np.ndarray:
    """Read one channel as a 2-D preview array shaped exactly ``target_shape``.

    ``tif.pages[i]`` yields a :class:`TiffPage`, which is a header object and not
    an array - it has to be materialised with ``asarray()`` before it can be
    sliced.  Indexing it directly is what made the segmentation PNG step fail
    with ``'TiffPage' object is not subscriptable``.

    A pyramid level is used whenever the file has one, so building a thumbnail
    of a 45k x 27k ROI does not materialise the full-resolution plane.  The
    result is resized to ``target_shape`` so it stays pixel-aligned with the
    strided segmentation mask it is drawn under.
    """
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    plane = None

    with tiff.TiffFile(ome_tiff_path) as tif:
        # --- preferred path: smallest pyramid level still >= the target ---
        try:
            series = tif.series[0]
            levels = list(getattr(series, "levels", []) or [])
            axes = getattr(series, "axes", "") or ""
            if len(levels) > 1 and "Y" in axes and "X" in axes:
                y_axis, x_axis = axes.index("Y"), axes.index("X")
                channel_axis = next((axes.index(name) for name in ("C", "I") if name in axes), None)
                for level in reversed(levels):
                    shape = level.shape
                    if shape[y_axis] < target_h or shape[x_axis] < target_w:
                        continue
                    selector = []
                    for position, name in enumerate(axes):
                        if name in ("Y", "X"):
                            selector.append(slice(None))
                        elif position == channel_axis:
                            selector.append(min(max(int(channel_index), 0), shape[position] - 1))
                        else:
                            selector.append(0)
                    try:
                        store = zarr.open(level.aszarr(), mode="r")
                        if not hasattr(store, "shape"):
                            store = store["0"] if "0" in store else store[list(store.array_keys())[0]]
                        plane = np.asarray(store[tuple(selector)])
                    except Exception:
                        plane = np.asarray(level.asarray()[tuple(selector)])
                    break
        except Exception:
            plane = None

        # --- fallback: page read ---
        if plane is None:
            page_index = int(channel_index) if 0 <= int(channel_index) < len(tif.pages) else 0
            array = tif.pages[page_index].asarray()
            if array.ndim == 3:
                # A single page holding the whole stack: index the channel axis.
                if array.shape[0] <= array.shape[-1]:
                    array = array[min(int(channel_index), array.shape[0] - 1)]
                else:
                    array = array[..., min(int(channel_index), array.shape[-1] - 1)]
            plane = np.asarray(array)

    plane = np.squeeze(plane)
    if plane.ndim != 2:
        plane = plane.reshape(plane.shape[-2], plane.shape[-1])
    if plane.shape != (target_h, target_w):
        plane = cv2.resize(
            plane.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA
        )
    return plane


def segment_image_cellpose(ome_tiff_path, output_dir, config, markers=None, dapi_channel='DAPI',
                          overwrite=False, progress_callback=None):
    """
    Complete segmentation pipeline for a single OME-TIFF image using Cellpose.
    
    Uses whole-image processing when no ROI mask is provided (fast path - lets Cellpose
    handle internal tiling). Uses manual tiling only when ROI masking is needed.
    
    Parameters:
    -----------
    ome_tiff_path : str
        Path to input OME-TIFF file
    output_dir : str
        Directory to save outputs
    config : SegmentationConfig
        Configuration object
    markers : list of str, optional
        List of marker channel names to use for segmentation
    dapi_channel : str
        Name of DAPI channel
    overwrite : bool
        Whether to overwrite existing outputs (default: False)
    progress_callback : callable, optional
        Callback function to report progress (current_step, total_steps, message)
        
    Returns:
    --------
    dict
        Dictionary containing paths to output files and segmentation results
    """
    start_time = datetime.now()
    input_image_name = os.path.splitext(os.path.basename(ome_tiff_path))[0]
    
    # Define steps
    total_steps = 5
    current_step = 0
    
    def update_progress(msg):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, msg)
        print(msg)

    # Check if outputs already exist
    update_progress(f"Checking existing outputs for {input_image_name}")
    if not overwrite:
        existing_outputs = check_segmentation_outputs_exist(ome_tiff_path, output_dir, config)
        if existing_outputs['all_exist']:
            print(f"Outputs already exist for {input_image_name}. Use overwrite=True to reprocess.")
            
            output_files = {}
            csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
            if os.path.exists(csv_path):
                output_files['features_csv'] = csv_path
                try:
                    output_files['features_df'] = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Warning: Could not load existing CSV: {e}")
            
            if config.save_geojson:
                geojson_path = os.path.join(output_dir, "geojson", f"{input_image_name}_cell_boundaries.geojson")
                if os.path.exists(geojson_path):
                    output_files['geojson'] = geojson_path
            
            if progress_callback:
                progress_callback(total_steps, total_steps, "Outputs already exist.")
            return output_files
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract channel information
    try:
        channel_names, _c_colors, _c_meta, _mpp = get_channel_info_any(ome_tiff_path)
    except Exception:
        channel_names = get_channel_names(ome_tiff_path)
    print(f"Channel names: {channel_names}")

    # Resolve DAPI channel
    dapi_index = -1
    if dapi_channel in channel_names:
        dapi_index = channel_names.index(dapi_channel)
    else:
        for i, nm in enumerate(channel_names):
            if isinstance(nm, str) and 'dapi' in nm.lower():
                dapi_index = i
                break
    if dapi_index < 0:
        raise ValueError(f"'{dapi_channel}' channel not found in image")
    
    # Determine marker indices
    if markers is None:
        marker_indices = [i for i, name in enumerate(channel_names) if i != dapi_index]
    else:
        marker_set = {m.lower(): m for m in markers if isinstance(m, str)}
        marker_indices = []
        for i, name in enumerate(channel_names):
            if i == dapi_index:
                continue
            if isinstance(name, str) and name.lower() in marker_set:
                marker_indices.append(i)
    
    print(f"DAPI channel index: {dapi_index}")
    print(f"Marker channel indices: {marker_indices}")
    
    # Initialize segmenter
    update_progress("Initializing segmentation")
    segmenter = CellposeSegmenter(config)
    
    # Decide whether to use fast whole-image path or tiled path
    use_tiled_processing = config.use_input_geojson and os.path.exists(config.input_geojson_path)
    
    if not use_tiled_processing:
        # FAST PATH: Load whole image, let Cellpose handle tiling internally
        return _segment_whole_image(
            ome_tiff_path, output_dir, config, segmenter,
            channel_names, dapi_index, marker_indices,
            input_image_name, progress_callback, update_progress, 
            current_step, total_steps, start_time
        )
    else:
        # TILED PATH: Manual tiling with ROI masking
        return _segment_with_tiles(
            ome_tiff_path, output_dir, config, segmenter,
            channel_names, dapi_index, marker_indices,
            input_image_name, progress_callback, update_progress,
            current_step, total_steps, start_time
        )


def _segment_whole_image(ome_tiff_path, output_dir, config, segmenter,
                         channel_names, dapi_index, marker_indices,
                         input_image_name, progress_callback, update_progress,
                         current_step, total_steps, start_time):
    """
    Fast path: process whole image at once, letting Cellpose handle internal tiling.
    This is much faster than manual tiling for GPU processing.
    """
    update_progress("Loading image channels...")
    
    # Load all channels at once
    with tiff.TiffFile(ome_tiff_path) as tif:
        (img_h, img_w), fullres_pages = _get_fullres_shape_and_page_indices(tif)
        print(f"Image dimensions: {img_h}x{img_w}")
        
        has_stack_page = any(getattr(p, 'ndim', 0) == 3 for p in tif.pages)
        use_stack_page = False
        stack_page_idx = 0
        if has_stack_page and not fullres_pages:
            use_stack_page = True
            stack_page_idx = 0
        
        name_to_page = _build_channel_page_map(tif, fullres_pages) if fullres_pages else {}
        
        # Build channel page indices
        channel_page_indices = []
        for i, nm in enumerate(channel_names):
            if use_stack_page:
                channel_page_indices.append(stack_page_idx)
                continue
            page_idx = name_to_page.get(nm) if isinstance(nm, str) else None
            if page_idx is None and i < len(fullres_pages):
                page_idx = fullres_pages[i]
            if page_idx is None:
                page_idx = i
            channel_page_indices.append(int(page_idx))
        
        # Read all channels
        all_images = []
        for ch_idx in range(len(channel_names)):
            if use_stack_page:
                page = tif.pages[stack_page_idx]
                if page.ndim == 3:
                    shp = page.shape
                    arr = page.asarray()
                    if shp[0] < shp[1] and shp[0] < shp[2]:
                        img = arr[ch_idx] if ch_idx < shp[0] else np.zeros((img_h, img_w), dtype=arr.dtype)
                    else:
                        img = arr[:, :, ch_idx] if ch_idx < shp[2] else np.zeros((img_h, img_w), dtype=arr.dtype)
                else:
                    img = np.zeros((img_h, img_w), dtype=np.uint16)
            else:
                page_idx = channel_page_indices[ch_idx]
                if page_idx < len(tif.pages):
                    page = tif.pages[page_idx]
                    if page.ndim == 2:
                        img = page.asarray()
                    elif page.ndim == 3:
                        arr = page.asarray()
                        img = arr[0] if arr.shape[0] < arr.shape[1] else arr[:, :, 0]
                    else:
                        img = np.zeros((img_h, img_w), dtype=np.uint16)
                else:
                    img = np.zeros((img_h, img_w), dtype=np.uint16)
            all_images.append(img)
        
        dapi_img = all_images[dapi_index]
        marker_imgs = [all_images[i] for i in marker_indices]
    
    # Prepare segmentation input
    merged_markers = np.max(marker_imgs, axis=0) if marker_imgs else np.zeros_like(dapi_img)
    image_for_segmentation = np.stack((
        merged_markers,
        np.zeros_like(merged_markers),
        dapi_img
    ), axis=-1)
    
    # Run segmentation - let Cellpose handle tiling internally (FAST!)
    update_progress("Running Cellpose segmentation...")
    
    def seg_progress(frac):
        if progress_callback:
            progress_callback(current_step + frac * 0.8, total_steps, 
                            f"Segmenting... {int(frac*100)}%")
    
    # Use Cellpose's internal tiling with tile=True (default)
    masks, flows, styles = segmenter.segment(
        image_for_segmentation,
        progress_callback=seg_progress,
        tile=True,  # Let Cellpose handle tiling
        tile_overlap=config.tile_overlap
    )
    
    print(f"Segmentation found {len(np.unique(masks)) - 1} cells")
    
    # Extract features
    update_progress("Extracting cell features...")
    features_df = extract_features(masks, all_images, channel_names)
    
    # Save outputs
    update_progress("Saving outputs...")
    output_files = {}
    
    # Save features CSV
    features_df, spatial_summary = apply_spatial_annotation(features_df, config, input_image_name)
    csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
    features_df.to_csv(csv_path, index=False)
    output_files['features_csv'] = csv_path
    output_files['features_df'] = features_df
    if spatial_summary is not None:
        output_files['spatial_summary'] = spatial_summary
    print(f"Features saved to: {csv_path}")
    
    # Save GeoJSON
    if config.save_geojson:
        geojson_subdir = os.path.join(output_dir, "geojson")
        os.makedirs(geojson_subdir, exist_ok=True)
        geojson_path = os.path.join(geojson_subdir, f"{input_image_name}_cell_boundaries.geojson")
        generate_geojson_from_mask(masks, geojson_path, dissolve_by_label=True)
        output_files['geojson'] = geojson_path
    
    # Save NPY mask
    if config.save_npy:
        npy_subdir = os.path.join(output_dir, "npy")
        os.makedirs(npy_subdir, exist_ok=True)
        npy_path = os.path.join(npy_subdir, f"{input_image_name}_mask.npy")
        try:
            np.save(npy_path, masks)
            output_files['mask_npy'] = npy_path
            print(f"Mask saved to: {npy_path}")
        except Exception as e:
            print(f"Warning: Could not save NPY mask: {e}")
    
    # Save PNG visualization
    if config.save_png:
        try:
            # Imported here rather than at module scope: matplotlib is only
            # needed for this optional preview and pulling it in at import
            # time would slow every segmentation run.
            import matplotlib.pyplot as plt

            # Imported here rather than at module scope: matplotlib is only
            # needed for this optional preview.

            png_subdir = os.path.join(output_dir, "png")
            os.makedirs(png_subdir, exist_ok=True)
            png_path = os.path.join(png_subdir, f"{input_image_name}_segmentation.png")
            factor = max(1, max(img_h, img_w) // 4000)
            small_mask = masks[::factor, ::factor]
            small_dapi = dapi_img[::factor, ::factor]
            
            plt_mask = np.ma.masked_where(small_mask == 0, small_mask)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(small_dapi, cmap='gray')
            ax.imshow(plt_mask, cmap='jet', alpha=0.5, interpolation='none')
            ax.axis('off')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            output_files['segmentation_png'] = png_path
            print(f"Visualization saved to: {png_path}")
        except Exception as e:
            print(f"Warning: Could not save PNG: {e}")
    
    output_files['features_df'] = features_df
    output_files['masks'] = masks
    
    elapsed_time = datetime.now() - start_time
    print(f"Segmentation completed in: {elapsed_time}")
    
    if progress_callback:
        progress_callback(total_steps, total_steps, "Complete!")
    
    return output_files


def _segment_with_tiles(ome_tiff_path, output_dir, config, segmenter,
                        channel_names, dapi_index, marker_indices,
                        input_image_name, progress_callback, update_progress,
                        current_step, total_steps, start_time):
    """
    Tiled path: process image in tiles with ROI masking support.
    Used when config.use_input_geojson is True.
    """
    # Open TIFF and get metadata
    with tiff.TiffFile(ome_tiff_path) as tif:
        (img_h, img_w), fullres_pages = _get_fullres_shape_and_page_indices(tif)
        print(f"Image dimensions: {img_h}x{img_w}")

        has_stack_page = any(getattr(p, 'ndim', 0) == 3 for p in tif.pages)
        use_stack_page = False
        stack_page_idx = 0
        if has_stack_page and not fullres_pages:
            use_stack_page = True
            stack_page_idx = 0

        name_to_page = _build_channel_page_map(tif, fullres_pages) if fullres_pages else {}

        channel_page_indices = []
        for i, nm in enumerate(channel_names):
            if use_stack_page:
                channel_page_indices.append(stack_page_idx)
                continue
            page_idx = name_to_page.get(nm) if isinstance(nm, str) else None
            if page_idx is None and i < len(fullres_pages):
                page_idx = fullres_pages[i]
            if page_idx is None:
                page_idx = i
            channel_page_indices.append(int(page_idx))
    
    # These variables are now available outside the with block
    # (Python allows this - the file is closed but variables persist)
    
    # Tiling configuration
    tile_size = config.tile_size
    overlap = config.tile_overlap
    overlap_px = int(tile_size * overlap)
    stride = tile_size - overlap_px
    
    # Calculate tiles
    all_tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            h = min(tile_size, img_h - y)
            w = min(tile_size, img_w - x)
            all_tiles.append((y, x, h, w))
            
    # Load the mask geometry once. Reopening and parsing the GeoJSON per tile
    # is prohibitively expensive for large images.
    tissue_geometry = None

    # Filter tiles by actual tissue coverage, not just a bounding-box touch.
    tiles = []
    print(f"Filtering tiles based on ROI: {config.input_geojson_path}")
    try:
        tissue_geometry = load_tissue_geometry(
        config.input_geojson_path,
        include_classes=getattr(config, "mask_class_names", None),
        exclude_classes=getattr(config, "mask_exclude_class_names", None),
    )
        print(f"ROI Bounds: {tissue_geometry.bounds}")
        
        for y, x, h, w in all_tiles:
            tile_box = Polygon([
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ])
            if not tissue_geometry.intersects(tile_box):
                continue
            tissue_area = tissue_geometry.intersection(tile_box).area
            if tissue_area / float(h * w) >= float(config.min_tissue_fraction):
                tiles.append((y, x, h, w))
        
        print(
            f"Filtered tiles: {len(tiles)}/{len(all_tiles)} meet the "
            f"{config.min_tissue_fraction:.3%} tissue-coverage threshold."
        )
    except Exception as e:
        print(f"Error filtering tiles by ROI: {e}. Processing all tiles.")
        tiles = all_tiles
    
    if not tiles:
        tiles = all_tiles
            
    print(f"Processing {len(tiles)} tiles (Size: {tile_size}, Overlap: {overlap_px}px)")
    
    # Initialize global mask (memmap) if needed
    global_mask = None
    temp_mask_dir = tempfile.mkdtemp()
    temp_mask_path = os.path.join(temp_mask_dir, "global_mask.dat")
    
    if config.save_npy or config.save_geojson or config.save_png:
        global_mask = np.memmap(temp_mask_path, dtype=np.uint32, mode='w+', shape=(img_h, img_w))
    
    # Initialize feature collection
    all_features = []
    current_max_label = 0

    # Track approximate written region to speed up GeoJSON export on large images
    bbox_has_content = False
    bbox_y0, bbox_y1 = img_h, 0
    bbox_x0, bbox_x1 = img_w, 0
    
    # Process tiles
    update_progress(f"Processing {len(tiles)} tiles...")
    
    # Granular progress for tiles
    base_step = current_step

    with tiff.TiffFile(ome_tiff_path) as tif:
        p0 = tif.pages[0]
        series_zarr = None
        per_channel_zarr = None
        series_axes = getattr(tif.series[0], "axes", "") if tif.series else ""
        channel_axis_present = any(axis in {"C", "I", "S"} for axis in series_axes)
        n_channels = len(channel_names)

        # There are two very different pyramidal layouts, and reading the wrong
        # one silently returns channel 0 for every requested channel:
        #   * one multi-channel series with a C axis -> select along that axis;
        #   * one series per channel followed by its levels (what the SPACEMAP
        #     registration and ROI writers produce) -> series[channel * levels].
        try:
            if channel_axis_present:
                series_zarr = _open_series_zarr_array(tif.series[0])
                print(f"Using Zarr region reads on the channel axis of series axes {series_axes}.")
            elif n_channels > 0 and len(tif.series) >= n_channels and len(tif.series) % n_channels == 0:
                levels_per_channel = len(tif.series) // n_channels
                per_channel_zarr = {}
                for channel_index in range(n_channels):
                    per_channel_zarr[channel_index] = _open_series_zarr_array(
                        tif.series[channel_index * levels_per_channel]
                    )
                print(
                    f"Using Zarr region reads across {n_channels} per-channel series "
                    f"({levels_per_channel} level(s) each)."
                )
            else:
                print("No channel-aware Zarr layout detected; using page reads.")
        except Exception as error:
            series_zarr = None
            per_channel_zarr = None
            print(f"Warning: Zarr region reads unavailable ({error}); using page reads.")

        def read_tile_safe(page_idx, channel_idx_in_page, y0, x0, hh, ww):
            if page_idx >= len(tif.pages):
                return np.zeros((hh, ww), dtype=p0.dtype)

            page = tif.pages[page_idx]

            if page.ndim == 2:
                arr = page.asarray()
                return arr[y0:y0 + hh, x0:x0 + ww]

            if page.ndim == 3:
                shp = page.shape
                if shp[0] < shp[1] and shp[0] < shp[2]:
                    if channel_idx_in_page >= shp[0]:
                        return np.zeros((hh, ww), dtype=page.dtype)
                    arr = page.asarray()
                    return arr[channel_idx_in_page, y0:y0 + hh, x0:x0 + ww]
                if shp[2] < shp[0] and shp[2] < shp[1]:
                    if channel_idx_in_page >= shp[2]:
                        return np.zeros((hh, ww), dtype=page.dtype)
                    arr = page.asarray()
                    return arr[y0:y0 + hh, x0:x0 + ww, channel_idx_in_page]
                if channel_idx_in_page >= shp[0]:
                    return np.zeros((hh, ww), dtype=page.dtype)
                arr = page.asarray()
                return arr[channel_idx_in_page, y0:y0 + hh, x0:x0 + ww]

            return np.zeros((hh, ww), dtype=page.dtype)

        def read_channel_tile(ch_idx: int, y0: int, x0: int, hh: int, ww: int):
            if per_channel_zarr is not None:
                array = per_channel_zarr.get(ch_idx)
                if array is not None:
                    return np.squeeze(np.asarray(array[y0:y0 + hh, x0:x0 + ww]))
            if series_zarr is not None and channel_axis_present:
                selector = []
                for axis_name in series_axes:
                    if axis_name == "Y":
                        selector.append(slice(y0, y0 + hh))
                    elif axis_name == "X":
                        selector.append(slice(x0, x0 + ww))
                    elif axis_name in {"C", "I", "S"}:
                        selector.append(ch_idx)
                    else:
                        selector.append(0)
                return np.squeeze(np.asarray(series_zarr[tuple(selector)]))
            if use_stack_page:
                return read_tile_safe(stack_page_idx, ch_idx, y0, x0, hh, ww)
            page_idx = channel_page_indices[ch_idx] if ch_idx < len(channel_page_indices) else ch_idx
            return read_tile_safe(page_idx, 0, y0, x0, hh, ww)

        for i, (y, x, h, w) in enumerate(tiles):
            fraction = i / len(tiles)
            if progress_callback:
                progress_callback(base_step + fraction * 0.9, total_steps, f"Processing tile {i+1}/{len(tiles)}")

            print(f"  Tile {i+1}/{len(tiles)}: y={y}, x={x}, h={h}, w={w}")

            if tissue_geometry is not None:
                tissue_mask = get_tissue_mask_for_tile_geometry((h, w), (y, x), tissue_geometry)
            else:
                tissue_mask = np.ones((h, w), dtype=np.uint8)
            crop_bounds = get_mask_crop_bounds(tissue_mask, config.tissue_crop_padding)
            if crop_bounds is None:
                # The tile passed the coverage filter, so an empty raster here
                # means the mask geometry failed to rasterize, not that the tile
                # is background. Say so loudly rather than dropping the tissue.
                print(
                    "    WARNING: tile passed the tissue-coverage filter but rasterized to "
                    "zero pixels; skipping. This indicates a mask geometry problem."
                )
                continue
            crop_y0, crop_y1, crop_x0, crop_x1 = crop_bounds
            crop_h = crop_y1 - crop_y0
            crop_w = crop_x1 - crop_x0
            source_y = y + crop_y0
            source_x = x + crop_x0
            tissue_mask_crop = tissue_mask[crop_y0:crop_y1, crop_x0:crop_x1]

            # Cellpose receives a single merged membrane/cytoplasm layer plus
            # DAPI. Full feature channels are read only after cells are found.
            dapi_tile = read_channel_tile(dapi_index, source_y, source_x, crop_h, crop_w)
            marker_tiles = [
                read_channel_tile(index, source_y, source_x, crop_h, crop_w)
                for index in marker_indices
            ]
            merged_markers = np.max(marker_tiles, axis=0) if marker_tiles else np.zeros_like(dapi_tile)
            merged_markers = merged_markers * tissue_mask_crop
            dapi_tile_masked = dapi_tile * tissue_mask_crop

            image_for_segmentation = np.stack((
                merged_markers,
                np.zeros_like(merged_markers),
                dapi_tile_masked
            ), axis=-1)

            # The outer loop already limits ROI work to config.tile_size. Let
            # Cellpose tile again within that outer tile using a memory-safe
            # bsize; `tile=False` previously forced one huge U-Net block.
            masks, _, _ = segmenter.segment(
                image_for_segmentation,
                tile=True,
                tile_overlap=config.tile_overlap,
                bsize=config.cellpose_internal_tile_size,
                batch_size=config.cellpose_batch_size,
            )

            props = regionprops_table(masks, properties=['label', 'centroid'])
            if len(props['label']) == 0:
                del masks, dapi_tile, marker_tiles, image_for_segmentation
                gc.collect()
                continue

            valid_labels = []

            y_min_valid = overlap_px // 2 if y > 0 else 0
            y_max_valid = h - (overlap_px // 2) if y + h < img_h else h
            x_min_valid = overlap_px // 2 if x > 0 else 0
            x_max_valid = w - (overlap_px // 2) if x + w < img_w else w

            for props_idx, label in enumerate(props['label']):
                cy = props['centroid-0'][props_idx] + crop_y0
                cx = props['centroid-1'][props_idx] + crop_x0

                if (y_min_valid <= cy < y_max_valid) and (x_min_valid <= cx < x_max_valid):
                    valid_labels.append(label)

            if not valid_labels:
                del masks, dapi_tile, marker_tiles, image_for_segmentation
                gc.collect()
                continue

            label_map = {old: new + current_max_label for new, old in enumerate(valid_labels, 1)}

            all_tile_images = [
                read_channel_tile(channel_index, source_y, source_x, crop_h, crop_w)
                for channel_index in range(len(channel_names))
            ]
            tile_features = extract_features(masks, all_tile_images, channel_names)
            tile_features = tile_features[tile_features['label'].isin(valid_labels)].copy()
            if tile_features.empty:
                del masks, dapi_tile, marker_tiles, all_tile_images, image_for_segmentation
                gc.collect()
                continue

            # Update labels and centroids
            tile_features['original_label'] = tile_features['label']
            tile_features['label'] = tile_features['label'].map(label_map)
            tile_features['y_centroid'] += source_y
            tile_features['x_centroid'] += source_x

            all_features.append(tile_features)

            # Update global mask
            if global_mask is not None:
                # Create a lookup array for fast relabeling
                max_val = masks.max()
                lookup = np.zeros(max_val + 1, dtype=np.uint32)
                for old, new in label_map.items():
                    lookup[old] = new

                relabeled_mask = lookup[masks]

                current_global_roi = global_mask[source_y:source_y + crop_h, source_x:source_x + crop_w]
                mask_indices = relabeled_mask > 0
                current_global_roi[mask_indices] = relabeled_mask[mask_indices]
                global_mask[source_y:source_y + crop_h, source_x:source_x + crop_w] = current_global_roi

                if mask_indices.any():
                    bbox_has_content = True
                    bbox_y0 = min(bbox_y0, source_y)
                    bbox_y1 = max(bbox_y1, source_y + crop_h)
                    bbox_x0 = min(bbox_x0, source_x)
                    bbox_x1 = max(bbox_x1, source_x + crop_w)

                if i % 10 == 0:
                    global_mask.flush()

            current_max_label += len(valid_labels)

            del masks, dapi_tile, marker_tiles, all_tile_images, image_for_segmentation
            gc.collect()

    # Merge results
    update_progress("Merging results")
    
    if all_features:
        final_features_df = pd.concat(all_features, ignore_index=True)
    else:
        final_features_df = pd.DataFrame()
        
    # Save outputs
    update_progress("Saving outputs")
    output_files = {}
    
    # Save features CSV
    final_features_df, spatial_summary = apply_spatial_annotation(final_features_df, config, input_image_name)
    csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
    final_features_df.to_csv(csv_path, index=False)
    output_files['features_csv'] = csv_path
    output_files['features_df'] = final_features_df
    if spatial_summary is not None:
        output_files['spatial_summary'] = spatial_summary
    print(f"Features saved to: {csv_path}")
    
    # Save segmentation mask as NPY
    if config.save_npy and global_mask is not None:
        npy_subdir = os.path.join(output_dir, "npy")
        os.makedirs(npy_subdir, exist_ok=True)
        npy_path = os.path.join(npy_subdir, f"{input_image_name}_mask.npy")
        try:
            np.save(npy_path, global_mask)
            output_files['mask_npy'] = npy_path
            print(f"Mask saved to: {npy_path}")
        except Exception as e:
            print(f"Warning: Could not save NPY mask (too large?): {e}")
    
    # Save segmentation visualization as PNG
    if config.save_png and global_mask is not None:
        try:
            png_subdir = os.path.join(output_dir, "png")
            os.makedirs(png_subdir, exist_ok=True)
            png_path = os.path.join(png_subdir, f"{input_image_name}_segmentation.png")
            factor = max(1, max(img_h, img_w) // 4000)
            small_mask = global_mask[::factor, ::factor]
            small_dapi = read_preview_plane(ome_tiff_path, dapi_index, small_mask.shape)

            plt_mask = np.ma.masked_where(small_mask == 0, small_mask)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(small_dapi, cmap='gray')
            ax.imshow(plt_mask, cmap='jet', alpha=0.5, interpolation='none')
            ax.axis('off')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            output_files['segmentation_png'] = png_path
            print(f"Segmentation visualization saved to: {png_path}")
        except Exception as e:
            print(f"Warning: Could not save PNG visualization: {e}")
    
    # Save cell boundaries as GeoJSON
    if config.save_geojson and global_mask is not None:
        geojson_subdir = os.path.join(output_dir, "geojson")
        os.makedirs(geojson_subdir, exist_ok=True)
        geojson_path = os.path.join(geojson_subdir, f"{input_image_name}_cell_boundaries.geojson")
        try:
            if bbox_has_content:
                y0 = max(0, int(bbox_y0))
                y1 = min(int(img_h), int(bbox_y1))
                x0 = max(0, int(bbox_x0))
                x1 = min(int(img_w), int(bbox_x1))
                print(f"GeoJSON export crop: y={y0}:{y1}, x={x0}:{x1}")
                generate_geojson_from_mask(
                    global_mask[y0:y1, x0:x1],
                    geojson_path,
                    offset_y=y0,
                    offset_x=x0,
                    dissolve_by_label=True,
                )
            else:
                generate_geojson_from_mask(global_mask, geojson_path, dissolve_by_label=True)
            output_files['geojson'] = geojson_path
        except Exception as e:
            print(f"Warning: Could not save GeoJSON: {e}")
    
    # Clean up memmap
    if global_mask is not None:
        del global_mask
        shutil.rmtree(temp_mask_dir, ignore_errors=True)
    
    # Store segmentation results (summary)
    output_files['features_df'] = final_features_df
    # We don't return full masks/flows/styles to avoid memory issues
    output_files['masks'] = None 
    
    elapsed_time = datetime.now() - start_time
    print(f"Segmentation completed in: {elapsed_time}")
    
    return output_files


def batch_segment_images(input_dir, output_dir, config, file_pattern="*.ome.tiff", 
                        markers=None, dapi_channel='DAPI', overwrite=False):
    """
    Batch process multiple OME-TIFF images for segmentation.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input OME-TIFF files
    output_dir : str
        Directory to save outputs
    config : SegmentationConfig
        Configuration object
    file_pattern : str
        File pattern to match (default: "*.ome.tiff")
    markers : list of str, optional
        List of marker channel names to use
    dapi_channel : str
        Name of DAPI channel
    overwrite : bool
        Whether to overwrite existing outputs (default: False)
        
    Returns:
    --------
    dict
        Dictionary mapping image names to their output files
    """
    import glob
    
    # Find all matching files
    search_pattern = os.path.join(input_dir, file_pattern)
    image_files = glob.glob(search_pattern)
    
    if not image_files:
        print(f"No files found matching pattern: {search_pattern}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    
    results = {}
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n--- Processing image {i}/{len(image_files)} ---")
        
        try:
            output_files = segment_image_cellpose(
                image_path, output_dir, config, markers, dapi_channel, overwrite
            )
            results[os.path.basename(image_path)] = output_files
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[os.path.basename(image_path)] = {'error': str(e)}
    
    print(f"\nBatch processing completed. Processed {len(results)} images.")
    return results


def check_segmentation_outputs_exist(input_image_path, output_dir, config):
    """
    Check if segmentation outputs already exist for an image.
    
    Parameters:
    -----------
    input_image_path : str
        Path to input image
    output_dir : str
        Output directory
    config : SegmentationConfig
        Configuration object
        
    Returns:
    --------
    dict
        Dictionary with existence status of each output type
    """
    input_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    # Check for required outputs
    csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
    csv_exists = os.path.exists(csv_path)
    
    # Check for optional outputs
    outputs = {
        'features_csv': csv_exists,
        'geojson': False,
        'mask_npy': False,
        'segmentation_png': False
    }
    
    if config.save_geojson:
        geojson_path = os.path.join(output_dir, "geojson", f"{input_image_name}_cell_boundaries.geojson")
        outputs['geojson'] = os.path.exists(geojson_path)
    
    if config.save_npy:
        npy_path = os.path.join(output_dir, "npy", f"{input_image_name}_mask.npy")
        outputs['mask_npy'] = os.path.exists(npy_path)
        
    if config.save_png:
        png_path = os.path.join(output_dir, "png", f"{input_image_name}_segmentation.png")
        outputs['segmentation_png'] = os.path.exists(png_path)
    
    # Consider outputs complete if all enabled outputs exist
    required_outputs = ['features_csv']
    if config.save_geojson:
        required_outputs.append('geojson')
    if config.save_npy:
        required_outputs.append('mask_npy')
    if config.save_png:
        required_outputs.append('segmentation_png')
    
    outputs['all_exist'] = all(outputs[key] for key in required_outputs)
    
    return outputs


# Placeholder for future segmentation methods
class DeepCellSegmenter:
    """Placeholder for DeepCell-based segmentation (future implementation)."""
    
    def __init__(self, config):
        self.config = config
        raise NotImplementedError("DeepCell segmentation not yet implemented")
    
    def segment(self, image_stack):
        pass


class StarDistSegmenter:
    """Placeholder for StarDist-based segmentation (future implementation)."""
    
    def __init__(self, config):
        self.config = config
        raise NotImplementedError("StarDist segmentation not yet implemented")
    
    def segment(self, image_stack):
        pass
