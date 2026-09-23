"""
Utility functions for SPACEMAP_V4 pipeline.

This module contains helper functions for image processing, metadata handling,
and file operations used throughout the SPACEMAP pipeline.
"""

import os
import json
import tifffile
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import itertools


def pairwise(iterable):
    """s -> (s0, s1), (s1, s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# Add pairwise to itertools for palom compatibility
itertools.pairwise = pairwise


def log_time(message, tab_id: str = None):
    """Log a message with timestamp. If tab_id is provided, only log if user is in that tab."""
    legacy = os.getenv("SPACEMAP_CONSOLE_LOGS")
    if legacy is not None:
        if legacy != "1":
            return
    elif os.getenv("SPACEMAP_TAB_LOGS", "1") != "1":
        return
    
    # If tab_id specified, only log if user is actually in that tab
    if tab_id:
        user_tab = os.environ.get("_spacemap_user_tab", "")
        if user_tab and user_tab != tab_id:
            return
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] {message}")


def hex_to_rgb(color_str):
    """
    Convert a hex color string (e.g. '#RRGGBB' or shorthand '#RGB') 
    or a comma-separated string into an RGB tuple.
    
    Args:
        color_str (str): Color string in hex format or comma-separated RGB values
        
    Returns:
        tuple: RGB values as (R, G, B)
    """
    color_str = color_str.strip()
    if color_str.startswith('#'):
        hex_str = color_str.lstrip('#')
        if len(hex_str) == 3:  # shorthand like #FFF
            hex_str = ''.join([c * 2 for c in hex_str])
        try:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            log_time(f"Failed to parse hex color '{color_str}': {e}")
            return (0, 0, 0)
    elif ',' in color_str:
        try:
            parts = color_str.split(',')
            return tuple(int(part.strip()) for part in parts)
        except Exception as e:
            log_time(f"Failed to parse comma-separated color '{color_str}': {e}")
            return (0, 0, 0)
    else:
        try:
            val = int(color_str)
            return (val, val, val)
        except Exception as e:
            log_time(f"Failed to parse color '{color_str}': {e}")
            return (0, 0, 0)


def check_file_exists(path, step_name, overwrite=False):
    """
    Check if file exists and log the result.
    
    Args:
        path (str): File path to check
        step_name (str): Name of the processing step for logging
        overwrite (bool): If True, returns False even if file exists (to force overwrite)
        
    Returns:
        bool: True if file exists and should be skipped, False otherwise
    """
    exists = os.path.exists(path)
    if exists:
        if overwrite:
            log_time(f"Overwriting existing output for {step_name}: {path}")
            return False
        else:
            log_time(f"Skipping {step_name} - Output already exists: {path}")
            return True
    return False


def rgb_to_color_int(rgb):
    """Convert an RGB-like color into a 32-bit signed integer.

    Accepts:
    - RGB tuple/list/np array: (R, G, B)
    - RGBA tuple/list: (R, G, B, A) (alpha is ignored)
    - Hex string: "#RRGGBB" or "RRGGBB"

    Notes
    -----
    The packed format used is: (R << 24) | (G << 16) | (B << 8).
    This matches the prior behavior in SPACEMAP.

    Parameters
    ----------
    rgb:
        Color specification.

    Returns
    -------
    int
        32-bit signed integer representation.
    """
    # Hex string support
    if isinstance(rgb, str):
        s = rgb.strip()
        if s.startswith('#'):
            s = s[1:]
        if len(s) == 6:
            try:
                R = int(s[0:2], 16)
                G = int(s[2:4], 16)
                B = int(s[4:6], 16)
            except Exception:
                R, G, B = 255, 255, 255
        else:
            R, G, B = 255, 255, 255
    else:
        # Sequence support (tuple/list/np array)
        try:
            seq = list(rgb)
        except Exception:
            seq = [255, 255, 255]
        if len(seq) >= 3:
            R, G, B = seq[0], seq[1], seq[2]
        else:
            R, G, B = 255, 255, 255

        try:
            R = int(R)
            G = int(G)
            B = int(B)
        except Exception:
            R, G, B = 255, 255, 255

    R = max(0, min(255, R))
    G = max(0, min(255, G))
    B = max(0, min(255, B))
    val = (R << 24) | (G << 16) | (B << 8)
    if val >= 2**31:
        val = val - 2**32
    return val


def create_ome_metadata(channel_names, channel_colors, SizeX, SizeY, pixel_type, n_channels, physical_size_x, physical_size_y, channel_metadata=None):
    """
    Create an OME-XML metadata string with channel names and colors.
    For each channel, we add a "Color" attribute computed via rgb_to_color_int,
    and add a nested <LightPath/> element. This mimics the QuPath-generated metadata.
    
    Args:
        channel_names (list): List of channel names
        channel_colors (list): List of RGB tuples for channel colors
        SizeX (int): Image width
        SizeY (int): Image height
        pixel_type (str): Pixel data type (e.g., 'uint16')
        n_channels (int): Number of channels
        physical_size_x (float): Physical pixel size in X dimension (microns)
        physical_size_y (float): Physical pixel size in Y dimension (microns)
        channel_metadata (list): Optional list of dicts containing 'dye' and 'exposure'
        
    Returns:
        str: OME-XML metadata string
    """
    ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    ome = ET.Element('OME', xmlns=ome_ns)
    image = ET.SubElement(ome, 'Image', ID='Image:0', Name='Merged Image')
    pixels = ET.SubElement(
        image,
        'Pixels',
        DimensionOrder='XYCZT',
        ID='Pixels:0',
        Type=pixel_type,
        SizeX=str(SizeX),
        SizeY=str(SizeY),
        SizeC=str(n_channels),
        SizeZ='1',
        SizeT='1',
        PhysicalSizeX=str(physical_size_x),
        PhysicalSizeY=str(physical_size_y)
    )

    for idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
        col_int = rgb_to_color_int(color) if color is not None else 0
        channel = ET.SubElement(
            pixels,
            'Channel',
            ID=f'Channel:0:{idx}',
            Name=name,
            SamplesPerPixel='1',
            Color=str(col_int)
        )
        
        # Add Metadata if available
        if channel_metadata and idx < len(channel_metadata):
            meta = channel_metadata[idx]
            if meta:
                dye = meta.get('dye', '-')
                exposure = meta.get('exposure', '-')
                excitation = meta.get('excitation_wavelength', '-')
                emission = meta.get('emission_wavelength', '-')
                
                # Add Fluor attribute (Standard OME)
                if dye and dye != '-':
                    channel.set('Fluor', str(dye))
                
                # Add Exposure as a custom attribute (Non-Standard but visible)
                if exposure and exposure != '-':
                    channel.set('Exposure', str(exposure))

                if excitation and excitation != '-':
                    channel.set('ExcitationWavelength', str(excitation))
                    channel.set('ExcitationWavelengthUnit', 'nm')

                if emission and emission != '-':
                    channel.set('EmissionWavelength', str(emission))
                    channel.set('EmissionWavelengthUnit', 'nm')

        ET.SubElement(channel, 'LightPath')
    
    for c in range(n_channels):
        ET.SubElement(
            pixels,
            'TiffData',
            FirstC=str(c),
            FirstZ='0',
            FirstT='0',
            IFD=str(c),
            PlaneCount='1'
        )
    
    return ET.tostring(ome, encoding='utf-8', method='xml').decode('utf-8')


def parse_xml(xml_string, elem_tag='Biomarker'):
    """
    Extract channel name, biomarker, color, dye, and exposure from XML metadata.
    
    Each QPTIFF page embeds per-channel XML with <Name> (the dye/fluorophore),
    <Biomarker>, <Color>, and <ExposureTime>.  We extract all of them so that
    every channel carries its own correct metadata — even when two channels
    share the same biomarker name but use different fluorophores.
    
    Args:
        xml_string (str): XML metadata string
        elem_tag (str): XML element tag to search for biomarker info
        
    Returns:
        dict: Dictionary with channel_name, biomarker, color, dye, and exposure
    """
    root = ET.fromstring(xml_string)
    name_element = root.find('Name')
    channel_name = name_element.text if name_element is not None else None
    biomarker_element = root.find(elem_tag)
    biomarker = biomarker_element.text if biomarker_element is not None else None
    color_element = root.find('Color')
    color = color_element.text if color_element is not None else None
    # Per-page dye (fluorophore filter name) and exposure
    exposure_element = root.find('ExposureTime')
    exposure_raw = exposure_element.text if exposure_element is not None else None
    try:
        exposure = str(int(float(exposure_raw) / 1000)) if exposure_raw else '-'
    except (ValueError, TypeError):
        exposure = '-'
    dye = channel_name if channel_name else '-'
    return {
        'channel_name': channel_name,
        'biomarker': biomarker,
        'color': color,
        'dye': dye,
        'exposure': exposure,
    }


def get_channel_names2(file_path, elem_tag='Biomarker'):
    """
    Extract channel names from a TIFF file's metadata.
    
    Args:
        file_path (str): Path to the TIFF file
        elem_tag (str): XML element tag to search for
        
    Returns:
        list: List of channel names
    """
    channel_names = []
    with tifffile.TiffFile(file_path) as tif:
        pages_to_iterate = tif.pages
        if tif.series:
            primary_series = max(tif.series, key=lambda s: np.prod(getattr(s, 'shape', (0, 0))))
            if primary_series:
                pages_to_iterate = primary_series.pages

        for page in pages_to_iterate:
            metadata = page.description
            try:
                info = parse_xml(metadata, elem_tag=elem_tag)
                channel_names.append(info['biomarker'])
            except Exception as e:
                log_time(f"Error parsing XML for a page: {e}")
    return channel_names


def remove_duplicates_preserve_order(channel_names):
    """
    Filter out None values but keep all names exactly as they are,
    including duplicates.  The original marker names from the file
    metadata are preserved without renaming.
    
    Args:
        channel_names (list): List of channel names
        
    Returns:
        list: Filtered list with None values removed
    """
    return [name for name in channel_names if name is not None]


def parse_qptiff(qptiff_path, elem_tag='Biomarker'):
    """
    Parse a qptiff file to extract channel (biomarker) names.
    
    Args:
        qptiff_path (str): Path to the qptiff file
        elem_tag (str): XML element tag to search for
        
    Returns:
        list: List of unique channel names
    """
    try:
        channel_names = get_channel_names2(qptiff_path, elem_tag=elem_tag)
        log_time(f"Initial channel names from {os.path.basename(qptiff_path)}: {channel_names}")
        channel_names = remove_duplicates_preserve_order(channel_names)
        log_time(f"Unique channel names: {channel_names}")
    except Exception as e:
        log_time("Couldn't load marker names from metadata; using fallback.")
        channel_names = []
    return channel_names


def load_geojson(geojson_path):
    """
    Load GeoJSON data from file.
    
    Args:
        geojson_path (str): Path to the GeoJSON file
        
    Returns:
        dict or None: GeoJSON data or None if loading failed
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        log_time(f"Error loading geojson {geojson_path}: {e}")
        return None


def extract_bboxes_from_geojson(geojson_data):
    """
    For each feature in the geojson, compute a bounding box as (min_x, min_y, max_x, max_y).
    Assumes geometries are Polygon or MultiPolygon.
    
    Args:
        geojson_data (dict): GeoJSON data
        
    Returns:
        list: List of bounding boxes as (min_x, min_y, max_x, max_y) tuples
    """
    bboxes = []
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        coords = []
        if geom.get('type') == 'Polygon':
            coords = np.array(geom.get('coordinates', [])[0])
        elif geom.get('type') == 'MultiPolygon':
            coords = np.array(geom.get('coordinates', [])[0][0])
        else:
            continue
        if coords.size == 0:
            continue
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        bboxes.append((int(min_x), int(min_y), int(max_x), int(max_y)))
    return bboxes


def save_roi_pyramidal_ome_tiff(layers, channel_names, channel_colors, output_path, physical_size):
    """
    Save ROI layers as a pyramidal OME-TIFF using the same metadata format as the merged image.
    Tile size: 256x256, compression = deflate (level 8), pyramid with 3 levels.
    
    Args:
        layers (list): List of image arrays
        channel_names (list): List of channel names
        channel_colors (list): List of RGB tuples for channel colors
        output_path (str): Output file path
        physical_size (float): Physical pixel size in microns
    """
    import cv2
    try:
        max_height = max(img.shape[0] for img in layers)
        max_width  = max(img.shape[1] for img in layers)
        tile_size = (256, 256)
        n_channels = len(layers)
        n_pyramid_levels = 3
        ome_xml = create_ome_metadata(
            channel_names=channel_names,
            channel_colors=channel_colors,
            SizeX=max_width,
            SizeY=max_height,
            pixel_type='uint16',
            n_channels=n_channels,
            physical_size_x=physical_size,
            physical_size_y=physical_size
        )
        with tifffile.TiffWriter(output_path, bigtiff=True) as tiff_writer:
            for c, img in enumerate(layers):
                pad_height = max_height - img.shape[0]
                pad_width = max_width - img.shape[1]
                padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant')
                tiff_writer.write(
                    padded_img,
                    subifds=n_pyramid_levels,
                    photometric='minisblack',
                    compression='deflate',
                    tile=tile_size,
                    description=ome_xml if c == 0 else None,
                    planarconfig='CONTIG',
                )
                prev_img = padded_img
                for i in range(n_pyramid_levels):
                    scale = 2 ** (i + 1)
                    new_width = max(1, max_width // scale)
                    new_height = max(1, max_height // scale)
                    downsampled_img = cv2.resize(prev_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    tiff_writer.write(
                        downsampled_img,
                        photometric='minisblack',
                        compression='deflate',
                        tile=tile_size,
                        planarconfig='CONTIG',
                    )
                    prev_img = downsampled_img
        log_time(f"ROI OME-TIFF saved: {output_path}")
    except Exception as e:
        log_time(f"Error saving ROI OME-TIFF: {e}")


def anchor_and_update_unified(existing, new_data, id_col='id', columns_to_update=None):
    """
    Anchor and update unified phenotyping file using ID-based merge.
    
    This function ensures that phenotype assignments are correctly linked to cells
    using the unique ID column, preventing misalignment issues that occur with
    positional merging.
    
    Parameters
    ----------
    existing : pd.DataFrame or None
        Existing unified data. If None, returns new_data as-is.
    new_data : pd.DataFrame
        New data with columns to add/update. Must contain id_col.
    id_col : str, default 'id'
        Column name to use for anchoring/merging.
    columns_to_update : list or None
        List of column names to update. If None, updates all columns in new_data
        except id_col.
        
    Returns
    -------
    pd.DataFrame
        Updated unified dataframe with properly anchored results.
    """
    
    # Validate inputs
    if id_col not in new_data.columns:
        raise ValueError(f"new_data missing required '{id_col}' column for anchoring")
    
    # If no existing data, return new_data
    if existing is None:
        return new_data.copy()
    
    # Validate existing data has id column
    if id_col not in existing.columns:
        raise ValueError(f"existing data missing required '{id_col}' column for anchoring")
    
    # Determine columns to update
    if columns_to_update is None:
        columns_to_update = [col for col in new_data.columns if col != id_col]
    
    # Perform ID-based merge
    existing = existing.set_index(id_col)
    new_data_indexed = new_data.set_index(id_col)
    
    # Update/add columns using ID-based alignment
    for col in columns_to_update:
        if col in new_data_indexed.columns:
            existing[col] = new_data_indexed[col]
    
    # Reset index to get id column back
    result = existing.reset_index()
    
    return result


def _wavelength_nm_to_hex(wavelength_nm) -> str:
    """Map a fluorescence emission wavelength (nm) to a display colour hex string."""
    try:
        wl = int(wavelength_nm)
    except (TypeError, ValueError):
        return '#FFFFFF'
    if wl < 420:
        return '#9900CC'   # UV / violet
    elif wl < 480:
        return '#0000FF'   # Blue  (DAPI ~460 nm)
    elif wl < 510:
        return '#00AAFF'   # Cyan
    elif wl < 560:
        return '#00FF00'   # Green (FITC ~520 nm)
    elif wl < 620:
        return '#FF8800'   # Orange (ATTO550 / Cy3 ~580 nm)
    elif wl < 720:
        return '#FF0000'   # Red (Cy5 ~670 nm)
    else:
        return '#AA0000'   # Deep red / IR (AF750 ~780 nm)


def _parse_perkinelmer_qpi_channels(description: str):
    """
    Parse channel names, colours, and MPP from a PerkinElmer-QPI-ImageDescription XML string.

    The format is produced by Vectra / Phenocycler Fusion instruments.  Each
    ``ScanBands-i`` element represents one acquired fluorescence channel; the
    active fluorophore is identified by ``ExcitedFilterPair/BandIndex`` indexing
    into the ``EmissionFilter/.../Bands-i`` list.

    Returns (names, hex_colors, metadata_list) or ([], [], []) on failure.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(description)
    except ET.ParseError:
        return [], [], []

    if 'PerkinElmer-QPI-ImageDescription' not in root.tag:
        return [], [], []

    names: list = []
    colors: list = []
    metadata_list: list = []

    for band in root.iter('ScanBands-i'):
        # Which spectral segment within the filter pair is active?
        exc_fp = band.find('.//ExcitedFilterPair')
        try:
            exc_band_idx = int((exc_fp.findtext('BandIndex') or '0').strip()) if exc_fp is not None else 0
        except (TypeError, ValueError):
            exc_band_idx = 0

        # Primary: get name + wavelength from EmissionFilter → Bands-i list
        em_filter = band.find('.//EmissionFilter')
        em_band_items = list(em_filter.iter('Bands-i')) if em_filter is not None else []
        em_names = [b.findtext('Name') for b in em_band_items if b.findtext('Name')]
        em_wls   = [b.findtext('HomeWavelength') for b in em_band_items if b.findtext('HomeWavelength')]

        # Fallback: excitation filter BandNames-i list
        exc_band_names = []
        if not em_names:
            src = band.find('.//ExcitationFilter') or band
            exc_band_names = [e.text.strip() for e in src.iter('BandNames-i') if e.text and e.text.strip()]

        if em_names and exc_band_idx < len(em_names):
            ch_name = em_names[exc_band_idx].strip()
            ch_wl   = em_wls[exc_band_idx] if exc_band_idx < len(em_wls) else None
        elif exc_band_names and exc_band_idx < len(exc_band_names):
            ch_name = exc_band_names[exc_band_idx]
            ch_wl   = em_wls[exc_band_idx] if exc_band_idx < len(em_wls) else None
        elif em_names:
            ch_name = em_names[0].strip()
            ch_wl   = em_wls[0] if em_wls else None
        else:
            ch_name = f'Channel {len(names)}'
            ch_wl   = None

        names.append(ch_name)
        colors.append(_wavelength_nm_to_hex(ch_wl))
        metadata_list.append({'dye': ch_name, 'emission_wl': str(ch_wl or '')})

    return names, colors, metadata_list


def get_channel_info_qptiff(file_path, elem_tag='Biomarker'):
    """
    Extract channel names, colors, and extra metadata (dye, exposure) from a QPTIFF file.
    
    Args:
        file_path (str): Path to the TIFF file
        elem_tag (str): XML element tag to search for
        
    Returns:
        tuple: (list of channel names, list of hex color strings, list of metadata dicts)
    """
    import re
    import json
    
    channel_names = []
    channel_colors = []
    channel_metadata = []
    
    metadata_map = {}

    with tifffile.TiffFile(file_path) as tif:
        # 1. Parse ScanProfile JSON for metadata mapping
        try:
            # Use getattr to avoid lint errors if stubs are incomplete
            desc = getattr(tif.pages[0], 'description', '')
            
            # Extract ScanProfile JSON
            # Look for <ScanProfile>...</ScanProfile>
            scan_profile_match = re.search(r"<ScanProfile>(.*?)</ScanProfile>", desc, re.DOTALL)
            if scan_profile_match:
                json_str = scan_profile_match.group(1).strip()
                # Clean potential CDATA or XML junk if present
                if json_str.startswith("<![CDATA["):
                    json_str = json_str.replace("<![CDATA[", "").replace("]]>", "")
                
                try:
                    data = json.loads(json_str)
                    
                    # Navigate to items: experimentDescription -> wells -> items
                    if 'experimentDescription' in data and 'wells' in data['experimentDescription']:
                        for well in data['experimentDescription']['wells']:
                            for item in well.get('items', []):
                                m_name = item.get('markerName')
                                if m_name:
                                    metadata_map[m_name] = {
                                        'dye': item.get('dye', '-'),
                                        'exposure': item.get('exposure', '-')
                                    }
                except json.JSONDecodeError:
                    # Fail silently or log debug only, as this is optional metadata
                    # log_time(f"Debug: ScanProfile content that failed parsing: {json_str[:100]}...")
                    pass
        except Exception as e:
            # log_time(f"Error parsing ScanProfile JSON: {e}")
            pass

        # 2. Iterate pages of the primary series to find actual channels in this image
        pages_to_iterate = tif.pages
        if tif.series:
            primary_series = max(tif.series, key=lambda s: np.prod(getattr(s, 'shape', (0, 0))))
            if primary_series:
                pages_to_iterate = primary_series.pages

        for page in pages_to_iterate:
            metadata = getattr(page, 'description', '')
            try:
                info = parse_xml(metadata, elem_tag=elem_tag)
                name = info.get('biomarker')
                
                if name:
                    channel_names.append(name)
                    
                    # Color
                    color_str = info.get('color')
                    if color_str:
                        try:
                            rgb = tuple(map(int, color_str.split(',')))
                            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                            channel_colors.append(hex_color)
                        except:
                            channel_colors.append("#FFFFFF")
                    else:
                        channel_colors.append("#FFFFFF")
                    
                    # Per-page dye and exposure from the XML (primary source)
                    page_dye = info.get('dye', '-')
                    page_exposure = info.get('exposure', '-')
                    # Fall back to ScanProfile map only if per-page is missing
                    if page_dye == '-' and name in metadata_map:
                        page_dye = metadata_map[name].get('dye', '-')
                    if page_exposure == '-' and name in metadata_map:
                        page_exposure = metadata_map[name].get('exposure', '-')
                    channel_metadata.append({'dye': page_dye, 'exposure': page_exposure})
            except Exception as e:
                # log_time(f"Error parsing XML for a page: {e}")
                pass
                
    # Fallback: PerkinElmer-QPI format (Vectra / Phenocycler Fusion) — no Biomarker tags
    if not channel_names:
        try:
            with tifffile.TiffFile(file_path) as tif:
                desc = getattr(tif.pages[0], 'description', '') or ''
            pe_names, pe_colors, pe_meta = _parse_perkinelmer_qpi_channels(desc)
            if pe_names:
                return pe_names, pe_colors, pe_meta
        except Exception:
            pass

    return channel_names, channel_colors, channel_metadata


def resolve_roi_label(properties, index=0):
    """Human-readable name for one ROI feature.

    QuPath writes an annotation's name to ``properties.name`` and its class to
    ``properties.classification.name``; other tools write ``properties.label``. Reading
    only one of these is why QuPath-authored ROIs used to show up as "ROI, ROI, ROI".
    """
    props = properties or {}

    for key in ('label', 'name'):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    classification = props.get('classification')
    if isinstance(classification, dict):
        value = classification.get('name')
        if value is not None and str(value).strip():
            return str(value).strip()

    value = props.get('id')
    if value is not None and str(value).strip():
        return str(value).strip()

    return f"ROI_{index + 1}"


def extract_rois_with_labels_from_geojson(geojson_data):
    """
    Extract bounding boxes and labels from GeoJSON data.
    
    Args:
        geojson_data (dict): GeoJSON data
        
    Returns:
        list: List of tuples ((min_x, min_y, max_x, max_y), label)
    """
    rois = []
    for i, feature in enumerate(geojson_data.get('features', [])):
        geom = feature.get('geometry', {})
        coords = []
        if geom.get('type') == 'Polygon':
            coords = np.array(geom.get('coordinates', [])[0])
        elif geom.get('type') == 'MultiPolygon':
            coords = np.array(geom.get('coordinates', [])[0][0])
        else:
            continue
        if coords.size == 0:
            continue
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        
        # Try to get label from properties
        props = feature.get('properties', {})
        label = resolve_roi_label(props, i)
        
        rois.append(((int(min_x), int(min_y), int(max_x), int(max_y)), str(label)))
    return rois


def extract_rois_with_geometry_from_geojson(geojson_data):
    """
    Extract bounding boxes, labels, and geometry from GeoJSON data.
    
    Args:
        geojson_data (dict): GeoJSON data
        
    Returns:
        list: List of tuples ((min_x, min_y, max_x, max_y), label, geometry)
              geometry is the raw geometry dictionary from the feature.
    """
    rois = []
    for i, feature in enumerate(geojson_data.get('features', [])):
        geom = feature.get('geometry', {})
        coords = []
        if geom.get('type') == 'Polygon':
            coords = np.array(geom.get('coordinates', [])[0])
        elif geom.get('type') == 'MultiPolygon':
            # For bounding box, we take the extent of all polygons
            all_coords = []
            for poly in geom.get('coordinates', []):
                all_coords.extend(poly[0])
            coords = np.array(all_coords)
        else:
            continue
            
        if coords.size == 0:
            continue
            
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        
        # Try to get label from properties
        props = feature.get('properties', {})
        label = resolve_roi_label(props, i)
        
        rois.append(((int(min_x), int(min_y), int(max_x), int(max_y)), str(label), geom))
    return rois

