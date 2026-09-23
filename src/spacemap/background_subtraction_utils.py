from typing import Tuple
from typing import Union, List, Optional, Any, Dict
"""
Utility functions for background subtraction in OME-TIFF images.
"""
import os
import re
import logging
import cv2
import tifffile
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from contextlib import contextmanager
import zarr
from spacemap.utils import log_time, create_ome_metadata, hex_to_rgb, get_channel_info_qptiff


SUPPORTED_IMAGE_EXTENSIONS = ('.ome.tiff', '.ome.tif', '.qptiff', '.tiff', '.tif')
RGB_CHANNEL_NAMES = ["HE_Red", "HE_Green", "HE_Blue"]
RGB_CHANNEL_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
KNOWN_DYE_WAVELENGTHS = {
    'DAPI': {'excitation': 405, 'emission': 461, 'color': (0, 0, 255)},
    'HOECHST': {'excitation': 405, 'emission': 461, 'color': (0, 0, 255)},
    'CY5': {'excitation': 649, 'emission': 670},
    'AF647': {'excitation': 650, 'emission': 668},
    'AF750': {'excitation': 749, 'emission': 775},
    'ATTO550': {'excitation': 554, 'emission': 576},
}


def io_worker_count() -> int:
    """Threads to hand tifffile for codec work.

    tifffile only parallelizes encode/decode when it is told how many threads it may
    use; its own ceiling is 32, so asking for more never helps.
    """
    return max(1, min(32, os.cpu_count() or 1))


def _log_active_tab(message: str):
    """Log only if user is in the current rendering tab, passing tab_id for filtering."""
    user_tab = os.environ.get("_spacemap_user_tab", "")
    active_tab = os.environ.get("_spacemap_active_tab", "")
    
    # If user_tab is set (GUI mode), only log if user is in this tab
    if user_tab:
        if active_tab and user_tab != active_tab:
            return  # User is in different tab, skip logging
        log_time(message, tab_id=active_tab if active_tab else None)
    elif active_tab:
        # GUI mode but user_tab not set yet - skip to avoid noise on fresh start
        return
    else:
        # Non-UI context (notebooks, scripts), always log
        log_time(message)


def normalize_image_stem(path_or_name: Union[str, Path]) -> str:
    name = Path(path_or_name).name
    lowered = name.lower()
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        if lowered.endswith(ext):
            return name[:-len(ext)]
    return Path(name).stem


def is_supported_image_file(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def _parse_ome_color(color_value) -> Tuple[int, int, int]:
    if color_value is None:
        return (255, 255, 255)
    try:
        packed = int(color_value)
    except (TypeError, ValueError):
        return (255, 255, 255)
    if packed < 0:
        packed += 2**32
    return ((packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF)


def _clean_metadata_value(value, default: str = '-') -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _format_numeric_metadata(value) -> str:
    if value in (None, '', '-'):
        return '-'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _clean_metadata_value(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip('0').rstrip('.')


def _extract_wavelength_from_text(text: Optional[str]) -> Tuple[str, Optional[str]]:
    if not text:
        return '-', None
    match = re.search(r'(?:^|[^A-Za-z0-9])((?:Argo|AF|ATTO)[_-]?\d{3,4}L?)(?:$|[^A-Za-z0-9])', text, re.IGNORECASE)
    if match:
        token = re.sub(r'\s+', '', match.group(1))
        digits = re.search(r'(\d{3,4})', token)
        return token, digits.group(1) if digits else None
    upper_text = text.upper()
    for dye_name, values in KNOWN_DYE_WAVELENGTHS.items():
        if dye_name in upper_text:
            emission = values.get('emission')
            return dye_name, str(int(emission)) if emission else None
    if re.fullmatch(r'AF\d+', upper_text):
        return upper_text, None
    return '-', None


def _wavelength_to_rgb(wavelength_nm: Union[float, int, str, None]) -> Tuple[int, int, int]:
    try:
        wavelength = float(wavelength_nm)
    except (TypeError, ValueError):
        return None

    anchors = [
        (405.0, (0, 0, 255)),
        (460.0, (0, 128, 255)),
        (500.0, (0, 255, 255)),
        (520.0, (0, 255, 0)),
        (550.0, (170, 255, 0)),
        (570.0, (255, 255, 0)),
        (590.0, (255, 170, 0)),
        (620.0, (255, 80, 0)),
        (680.0, (255, 0, 0)),
        (760.0, (255, 0, 180)),
        (875.0, (255, 200, 255)),
    ]

    wavelength = min(max(wavelength, anchors[0][0]), anchors[-1][0])
    for (left_wavelength, left_color), (right_wavelength, right_color) in zip(anchors, anchors[1:]):
        if wavelength <= right_wavelength:
            if right_wavelength == left_wavelength:
                return left_color
            fraction = (wavelength - left_wavelength) / (right_wavelength - left_wavelength)
            return tuple(
                int(round(left_color[channel_index] + fraction * (right_color[channel_index] - left_color[channel_index])))
                for channel_index in range(3)
            )

    return anchors[-1][1]


def _infer_channel_metadata(name: str) -> Dict[str, Any]:
    normalized_name = _clean_metadata_value(name, default='Unknown')
    upper_name = normalized_name.upper()

    if normalized_name in RGB_CHANNEL_NAMES:
        rgb_index = RGB_CHANNEL_NAMES.index(normalized_name)
        return {
            'dye': normalized_name,
            'exposure': '-',
            'excitation_wavelength': '-',
            'emission_wavelength': '-',
            'color': RGB_CHANNEL_COLORS[rgb_index],
        }

    if 'HOECHST' in upper_name:
        values = KNOWN_DYE_WAVELENGTHS['HOECHST']
        return {
            'dye': 'Hoechst',
            'exposure': '-',
            'excitation_wavelength': str(values['excitation']),
            'emission_wavelength': str(values['emission']),
            'color': values['color'],
        }

    if 'DAPI' in upper_name:
        values = KNOWN_DYE_WAVELENGTHS['DAPI']
        return {
            'dye': 'DAPI',
            'exposure': '-',
            'excitation_wavelength': str(values['excitation']),
            'emission_wavelength': str(values['emission']),
            'color': values['color'],
        }

    if re.fullmatch(r'AF\d+', upper_name):
        return {
            'dye': upper_name,
            'exposure': '-',
            'excitation_wavelength': '-',
            'emission_wavelength': '-',
            'color': (192, 192, 192),
        }

    dye_name, emission_nm = _extract_wavelength_from_text(normalized_name)
    if emission_nm is not None:
        inferred_color = _wavelength_to_rgb(emission_nm)
        excitation_nm = '-'
        if dye_name.upper() in KNOWN_DYE_WAVELENGTHS:
            known_values = KNOWN_DYE_WAVELENGTHS[dye_name.upper()]
            excitation_nm = _format_numeric_metadata(known_values.get('excitation'))
            if inferred_color is None and known_values.get('color') is not None:
                inferred_color = known_values['color']
        return {
            'dye': dye_name,
            'exposure': '-',
            'excitation_wavelength': excitation_nm,
            'emission_wavelength': emission_nm,
            'color': inferred_color or (255, 255, 255),
        }

    for dye_name, values in KNOWN_DYE_WAVELENGTHS.items():
        if dye_name in upper_name:
            return {
                'dye': dye_name,
                'exposure': '-',
                'excitation_wavelength': _format_numeric_metadata(values.get('excitation')),
                'emission_wavelength': _format_numeric_metadata(values.get('emission')),
                'color': values.get('color') or _wavelength_to_rgb(values.get('emission')) or (255, 255, 255),
            }

    return {
        'dye': '-',
        'exposure': '-',
        'excitation_wavelength': '-',
        'emission_wavelength': '-',
        'color': (255, 255, 255),
    }


def _build_channel_record(
    name: str,
    explicit_color: Tuple[int, int, int],
    explicit_metadata: Dict[str, object],
    source: str,
) -> Dict[str, object]:
    inferred = _infer_channel_metadata(name)
    explicit_metadata = explicit_metadata or {}

    color = explicit_color if explicit_color is not None else inferred['color']
    metadata = {
        'dye': _clean_metadata_value(explicit_metadata.get('dye')),
        'exposure': _clean_metadata_value(explicit_metadata.get('exposure')),
        'excitation_wavelength': _format_numeric_metadata(explicit_metadata.get('excitation_wavelength')),
        'emission_wavelength': _format_numeric_metadata(explicit_metadata.get('emission_wavelength')),
        'color_source': 'embedded' if explicit_color is not None else 'inferred',
        'metadata_source': source,
    }

    if metadata['dye'] == '-' and inferred['dye'] != '-':
        metadata['dye'] = _clean_metadata_value(inferred['dye'])
    if metadata['excitation_wavelength'] == '-' and inferred['excitation_wavelength'] != '-':
        metadata['excitation_wavelength'] = _clean_metadata_value(inferred['excitation_wavelength'])
    if metadata['emission_wavelength'] == '-' and inferred['emission_wavelength'] != '-':
        metadata['emission_wavelength'] = _clean_metadata_value(inferred['emission_wavelength'])

    return {
        'name': name,
        'color': color,
        'metadata': metadata,
    }


def _get_ome_pixels_element(ome_xml: str):
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    root = ET.fromstring(ome_xml)
    pixels_elem = root.find('.//ome:Pixels', namespace)
    if pixels_elem is None:
        pixels_elem = root.find('.//Pixels')
    return root, pixels_elem, namespace


def _build_generic_channel_records(shape, axes) -> List[Dict[str, object]]:
    try:
        dims = tuple(int(dim) for dim in shape)
    except Exception:
        return []

    channel_count = None
    channel_axis = None
    if axes:
        channel_axis, _channel_axis_name = _channel_axis_from_axes(axes)
        if channel_axis is not None and channel_axis < len(dims):
            candidate = int(dims[channel_axis])
            if candidate > 0:
                channel_count = candidate

    if channel_count is None and len(dims) == 3:
        if int(dims[0]) <= 16:
            channel_count = int(dims[0])
        elif int(dims[-1]) <= 16:
            channel_count = int(dims[-1])

    if channel_count is None and len(dims) == 2:
        # A single-plane 2D image has exactly one channel and no channel axis to find.
        # Without this the caller reports "Could not extract channel info" for every
        # plain grayscale TIFF — such as the registration reference written by
        # extract_dapi_channel, which is one DAPI plane with no OME or QPTIFF metadata.
        channel_count = 1

    if channel_count is None or channel_count <= 0:
        return []

    if channel_count == 3:
        return [
            {
                'name': name,
                'color': color,
                'metadata': {
                    'dye': name,
                    'exposure': '-',
                    'excitation_wavelength': '-',
                    'emission_wavelength': '-',
                    'color_source': 'synthetic-rgb',
                    'metadata_source': 'generic-tiff-rgb',
                },
            }
            for name, color in zip(RGB_CHANNEL_NAMES, RGB_CHANNEL_COLORS)
        ]

    return [
        _build_channel_record(
            name=f'Channel_{channel_index}',
            explicit_color=None,
            explicit_metadata=None,
            source='generic-tiff',
        )
        for channel_index in range(channel_count)
    ]


def _coerce_positive_mpp(value) -> float | None:
    try:
        mpp = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mpp) or mpp <= 0:
        return None
    return mpp


def _ome_length_unit_to_microns(unit_value) -> float | None:
    if unit_value is None:
        return 1.0

    normalized = str(unit_value).strip().lower()
    normalized = normalized.replace('µ', 'u')
    normalized = normalized.replace('micro', 'u')

    unit_map = {
        'um': 1.0,
        'micrometer': 1.0,
        'micrometers': 1.0,
        'micrometre': 1.0,
        'micrometres': 1.0,
        'nm': 1e-3,
        'nanometer': 1e-3,
        'nanometers': 1e-3,
        'nanometre': 1e-3,
        'nanometres': 1e-3,
        'mm': 1e3,
        'millimeter': 1e3,
        'millimeters': 1e3,
        'millimetre': 1e3,
        'millimetres': 1e3,
        'cm': 1e4,
        'centimeter': 1e4,
        'centimeters': 1e4,
        'centimetre': 1e4,
        'centimetres': 1e4,
        'm': 1e6,
        'meter': 1e6,
        'meters': 1e6,
        'metre': 1e6,
        'metres': 1e6,
        'in': 25400.0,
        'inch': 25400.0,
        'inches': 25400.0,
    }
    return unit_map.get(normalized)


def _extract_mpp_from_description(description: Optional[str]) -> float | None:
    text = str(description or '')
    if not text:
        return None

    patterns = (
        r'\bMPP\s*=\s*([0-9]+(?:\.[0-9]+)?)',
        r'\bMicronsPerPixel\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        r'\bopenslide\.mpp-x\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        r'\bmpp-x\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        r'<PixelSizeMicrons>\s*([0-9]+(?:\.[0-9]+)?)\s*</PixelSizeMicrons>',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            mpp = _coerce_positive_mpp(match.group(1))
            if mpp is not None:
                return mpp
    return None


def _resolution_unit_to_microns(resolution_unit) -> float | None:
    if resolution_unit is None:
        return 25400.0

    if hasattr(resolution_unit, 'name'):
        normalized = str(resolution_unit.name).strip().lower()
    else:
        normalized = str(resolution_unit).strip().lower()

    unit_map = {
        '2': 25400.0,
        'inch': 25400.0,
        'inches': 25400.0,
        '3': 10000.0,
        'centimeter': 10000.0,
        'centimeters': 10000.0,
        'centimetre': 10000.0,
        'centimetres': 10000.0,
    }
    return unit_map.get(normalized)


def _pixels_per_unit_from_resolution(resolution) -> float | None:
    if resolution is None:
        return None

    try:
        values = list(resolution)
    except TypeError:
        values = [resolution]

    numeric_values = []
    for value in values[:2]:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value) and numeric_value > 0:
            numeric_values.append(numeric_value)

    if not numeric_values:
        return None
    return float(numeric_values[0] if len(numeric_values) == 1 else np.mean(numeric_values))


def _extract_mpp_from_tif(tif: tifffile.TiffFile, default: float | None = None) -> float | None:
    try:
        if tif.ome_metadata:
            _root, pixels_elem, _namespace = _get_ome_pixels_element(tif.ome_metadata)
            if pixels_elem is not None:
                value = pixels_elem.attrib.get('PhysicalSizeX')
                if value not in (None, ''):
                    scale = _ome_length_unit_to_microns(pixels_elem.attrib.get('PhysicalSizeXUnit'))
                    physical_size = _coerce_positive_mpp(value)
                    if physical_size is not None and scale is not None:
                        return physical_size * scale
    except Exception:
        pass

    try:
        page = tif.pages[0]
        description_mpp = _extract_mpp_from_description(getattr(page, 'description', None))
        if description_mpp is not None:
            return description_mpp
    except Exception:
        pass

    try:
        page = tif.pages[0]
        res = getattr(page, 'resolution', None)
        px_per_unit = _pixels_per_unit_from_resolution(res)
        if px_per_unit is not None:
            resolution_unit = page.tags.get('ResolutionUnit')
            unit_scale = _resolution_unit_to_microns(
                resolution_unit.value if resolution_unit is not None else None
            )
            if unit_scale is not None:
                return unit_scale / px_per_unit
    except Exception:
        pass

    return default


@contextmanager
def _quiet_tifffile_logging(level: int = logging.ERROR):
    logger = logging.getLogger('tifffile')
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def get_image_mpp(image_path: str, default: float | None = None) -> Dict[str, float | None]:
    try:
        with _quiet_tifffile_logging():
            with tifffile.TiffFile(image_path) as tif:
                return {'codex': _extract_mpp_from_tif(tif, default=default)}
    except Exception:
        return {'codex': default}


def _get_primary_series(tif: tifffile.TiffFile):
    if not tif.series:
        return None
    return max(tif.series, key=lambda series: int(np.prod(getattr(series, 'shape', (0,)))))


def _channel_axis_from_axes(axes: Optional[str]) -> Tuple[int | None, Optional[str]]:
    axes = axes or ''
    for axis_name in ('C', 'I', 'S'):
        if axis_name in axes:
            return axes.index(axis_name), axis_name
    return None, None


def _get_spatial_shape(shape, axes: Optional[str] = None) -> Tuple[int, int]:
    axes = axes or ''
    if axes and 'Y' in axes and 'X' in axes:
        return int(shape[axes.index('Y')]), int(shape[axes.index('X')])
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    if len(shape) == 1:
        return int(shape[0]), 1
    return 0, 0


def _get_preview_level_index(series, max_dim: int = 1400) -> int:
    levels = list(getattr(series, 'levels', [])) or [series]
    for idx, level in enumerate(levels):
        h, w = _get_spatial_shape(level.shape, getattr(level, 'axes', getattr(series, 'axes', '')))
        if max(h, w) <= max_dim:
            return idx
    return len(levels) - 1


def _ome_channel_ifd_map(ome_xml: Optional[str]) -> Dict[int, int]:
    if not ome_xml:
        return {}
    try:
        _root, pixels_elem, namespace = _get_ome_pixels_element(ome_xml)
    except Exception:
        return {}
    if pixels_elem is None:
        return {}

    mapping = {}
    tiff_datas = pixels_elem.findall('ome:TiffData', namespace) or pixels_elem.findall('TiffData')
    for elem in tiff_datas:
        try:
            mapping[int(elem.attrib.get('FirstC', '0'))] = int(elem.attrib['IFD'])
        except (KeyError, TypeError, ValueError):
            continue
    return mapping


def _fullres_page_indices(tif: tifffile.TiffFile) -> List[int]:
    max_hw = (0, 0)
    for page in tif.pages:
        if getattr(page, 'ndim', 0) == 2:
            h, w = map(int, page.shape)
            if h * w > max_hw[0] * max_hw[1]:
                max_hw = (h, w)
    if max_hw == (0, 0) and tif.pages:
        max_hw = _get_spatial_shape(tif.pages[0].shape, getattr(tif.pages[0], 'axes', None))
    return [
        idx for idx, page in enumerate(tif.pages)
        if getattr(page, 'ndim', 0) == 2 and tuple(map(int, page.shape)) == max_hw
    ]


def _read_channel_from_series(series, channel_index: int, level_index: int = 0, region: Tuple[int, int, int, int] = None) -> np.ndarray:
    axes = getattr(series, 'axes', '') or ''
    channel_axis, _channel_axis_name = _channel_axis_from_axes(axes)
    if channel_axis is None:
        raise ValueError(f"Series does not expose a channel axis: axes={axes!r}")

    zarr_array = zarr.open(series.aszarr(level=level_index), mode='r')
    selector = []
    for axis_idx, axis_name in enumerate(axes):
        if axis_name == 'Y':
            selector.append(slice(region[0], region[1]) if region else slice(None))
        elif axis_name == 'X':
            selector.append(slice(region[2], region[3]) if region else slice(None))
        elif axis_idx == channel_axis:
            selector.append(channel_index)
        else:
            selector.append(0)
    return np.asarray(zarr_array[tuple(selector)])


def _read_full_channel_via_pages(series, channel_index: int, level_index: int = 0) -> Optional[np.ndarray]:
    """Decode one whole channel plane through tifffile's multithreaded codec path.

    The zarr view used by :func:`_read_channel_from_series` decodes tiles serially. When
    the level stores exactly one page per channel — the layout of QPTIFF and of the
    OME-TIFFs written by this pipeline — the page can be decoded directly with several
    threads instead. Returns ``None`` for any other layout so the caller falls back.
    """
    try:
        levels = list(getattr(series, 'levels', []) or []) or [series]
        if not 0 <= level_index < len(levels):
            return None
        level = levels[level_index]
        axes = getattr(level, 'axes', '') or getattr(series, 'axes', '') or ''
        shape = tuple(int(dim) for dim in (getattr(level, 'shape', ()) or ()))
        if axes != 'CYX' or len(shape) != 3 or not 0 <= channel_index < shape[0]:
            return None
        pages = list(getattr(level, 'pages', []) or [])
        if len(pages) != shape[0]:
            return None
        page = pages[channel_index]
        if getattr(page, 'ndim', 0) != 2 or tuple(int(dim) for dim in page.shape) != shape[1:]:
            return None
        return page.asarray(maxworkers=io_worker_count())
    except Exception:
        return None


def _read_channel_image_from_tif(
    tif: tifffile.TiffFile,
    channel_index: int,
    region: Tuple[int, int, int, int] = None,
    level_index: int = 0,
) -> np.ndarray:
    series = _get_primary_series(tif)
    if series is not None:
        axes = getattr(series, 'axes', '') or ''
        channel_axis, _channel_axis_name = _channel_axis_from_axes(axes)
        if channel_axis is not None and len(getattr(series, 'shape', ())) >= 3:
            if region is None:
                plane = _read_full_channel_via_pages(series, channel_index, level_index=level_index)
                if plane is not None:
                    return plane
            return np.squeeze(_read_channel_from_series(series, channel_index, level_index=level_index, region=region))

    page_index = _ome_channel_ifd_map(tif.ome_metadata).get(channel_index)
    if page_index is None:
        fullres_pages = _fullres_page_indices(tif)
        if channel_index < len(fullres_pages):
            page_index = fullres_pages[channel_index]
        else:
            page_index = min(channel_index, max(len(tif.pages) - 1, 0))

    page = tif.pages[page_index]
    data = np.asarray(page.asarray())
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            data = data[channel_index]
        elif data.shape[-1] in (3, 4):
            data = data[..., channel_index]
    if region is not None and data.ndim >= 2:
        y1, y2, x1, x2 = region
        data = data[y1:y2, x1:x2]
    return np.squeeze(np.asarray(data))


def read_channel_image(
    image_path: str,
    channel_index: int,
    region: Tuple[int, int, int, int] = None,
    level_index: int = 0,
) -> np.ndarray:
    with tifffile.TiffFile(image_path) as tif:
        return _read_channel_image_from_tif(tif, channel_index, region=region, level_index=level_index)


def read_channel_from_open_tif(
    tif: tifffile.TiffFile,
    channel_index: int,
    region: Tuple[int, int, int, int] = None,
    level_index: int = 0,
) -> np.ndarray:
    """Read a channel, or a region of one, from an already-open ``TiffFile``.

    Opening a TIFF forces a full series scan, which on a 52-channel pyramidal stack
    costs well over a second — far more than decoding a single ROI crop. Callers pulling
    many channels out of one image should open the file once and reuse the handle rather
    than calling :func:`read_channel_image` per channel.
    """
    return _read_channel_image_from_tif(tif, channel_index, region=region, level_index=level_index)


def load_preview_channels_compatible(file_path: str, indices_to_load: List[int], max_dim: int = 1000):
    loaded_data = {}
    full_shape = (0, 0)
    is_rgb = False

    with tifffile.TiffFile(file_path) as tif:
        series = _get_primary_series(tif)
        if series is not None:
            full_shape = _get_spatial_shape(series.shape, getattr(series, 'axes', None))
            axes = getattr(series, 'axes', '') or ''
            channel_axis, channel_axis_name = _channel_axis_from_axes(axes)

            if channel_axis is not None and len(getattr(series, 'shape', ())) >= 3:
                preview_level = _get_preview_level_index(series, max_dim=max_dim)
                preview_store = zarr.open(series.aszarr(level=preview_level), mode='r')
                if channel_axis_name == 'S' and series.shape[channel_axis] in (3, 4):
                    rgb = np.asarray(preview_store)
                    if channel_axis == 0:
                        rgb = np.moveaxis(rgb, 0, -1)
                    return {'rgb': np.squeeze(rgb)}, full_shape, True

                for idx in indices_to_load:
                    selector = []
                    for axis_idx, axis_name in enumerate(axes):
                        if axis_name in ('Y', 'X'):
                            selector.append(slice(None))
                        elif axis_idx == channel_axis:
                            selector.append(idx)
                        else:
                            selector.append(0)
                    loaded_data[idx] = np.asarray(preview_store[tuple(selector)])
                return loaded_data, full_shape, is_rgb

        if tif.pages:
            first_page = np.asarray(tif.pages[0].shape)
            if len(first_page) == 3 and (first_page[0] in (3, 4) or first_page[-1] in (3, 4)):
                rgb = np.asarray(tif.pages[0].asarray())
                if rgb.shape[0] in (3, 4):
                    rgb = np.moveaxis(rgb, 0, -1)
                scale = max(rgb.shape[0], rgb.shape[1]) // max_dim
                if scale > 1:
                    rgb = rgb[::scale, ::scale]
                return {'rgb': rgb}, _get_spatial_shape(rgb.shape), True

            scale = max(full_shape) // max_dim if max(full_shape) else 1
            if scale < 1:
                scale = 1
            for idx in indices_to_load:
                plane = _read_channel_image_from_tif(tif, idx)
                if scale > 1 and plane.ndim == 2:
                    plane = plane[::scale, ::scale]
                loaded_data[idx] = plane

    return loaded_data, full_shape, is_rgb


def get_channel_records(image_path: str) -> Tuple[List[Dict[str, object]], Dict[str, float | None]]:
    """Return normalized per-channel metadata for OME-TIFF and QPTIFF inputs."""
    _log_active_tab(f"Discovering channels in: {Path(image_path).name}")

    def _make_records_unique(recs):
        """Pass through records without renaming — preserve original marker names."""
        return recs

    mpp = {'codex': None}
    try:
        with tifffile.TiffFile(image_path) as tif:
            mpp['codex'] = _extract_mpp_from_tif(tif, default=mpp['codex'])

            if tif.ome_metadata:
                _root, pixels_elem, namespace = _get_ome_pixels_element(tif.ome_metadata)
                if pixels_elem is not None:
                    channel_elems = pixels_elem.findall('.//ome:Channel', namespace)
                    if not channel_elems:
                        channel_elems = pixels_elem.findall('.//Channel')
                    size_c = int(pixels_elem.attrib.get('SizeC', len(channel_elems) or 1))

                    if size_c == 3 and (len(channel_elems) <= 1 or all(not ch.attrib.get('Name') for ch in channel_elems)):
                        records = [
                            {
                                'name': name,
                                'color': color,
                                'metadata': {
                                    'dye': name,
                                    'exposure': '-',
                                    'excitation_wavelength': '-',
                                    'emission_wavelength': '-',
                                    'color_source': 'synthetic-rgb',
                                    'metadata_source': 'ome-rgb-fallback',
                                },
                            }
                            for name, color in zip(RGB_CHANNEL_NAMES, RGB_CHANNEL_COLORS)
                        ]
                        _log_active_tab(f"Found {len(records)} RGB channels via OME-XML.")
                        return records, mpp

                    if channel_elems or size_c > 0:
                        records = []
                        total_channels = max(size_c, len(channel_elems))
                        for i in range(total_channels):
                            ch = channel_elems[i] if i < len(channel_elems) else None
                            name = ch.attrib.get('Name') if ch is not None else None
                            color_attr = ch.attrib.get('Color') if ch is not None else None
                            explicit_color = _parse_ome_color(color_attr) if color_attr not in (None, '') else None
                            explicit_metadata = {
                                'dye': ch.attrib.get('Fluor') if ch is not None else None,
                                'exposure': ch.attrib.get('Exposure') if ch is not None else None,
                                'excitation_wavelength': ch.attrib.get('ExcitationWavelength') if ch is not None else None,
                                'emission_wavelength': ch.attrib.get('EmissionWavelength') if ch is not None else None,
                            }
                            records.append(
                                _build_channel_record(
                                    name=name if name else f'Channel_{i}',
                                    explicit_color=explicit_color,
                                    explicit_metadata=explicit_metadata,
                                    source='ome',
                                )
                            )

                        _log_active_tab(f"Found {len(records)} channels via OME-XML.")
                        return _make_records_unique(records), mpp

            _log_active_tab("OME-XML not found or incomplete. Trying QPTIFF metadata...")
            q_names, q_colors, q_meta = get_channel_info_qptiff(image_path)
            if q_names:
                records = []
                for i, name in enumerate(q_names):
                    color_value = q_colors[i] if i < len(q_colors) else None
                    explicit_color = None
                    if color_value not in (None, '', '-'):
                        explicit_color = hex_to_rgb(color_value) if isinstance(color_value, str) else tuple(color_value)
                    metadata = q_meta[i] if i < len(q_meta) else {}
                    records.append(
                        _build_channel_record(
                            name=name,
                            explicit_color=explicit_color,
                            explicit_metadata=metadata,
                            source='qptiff',
                        )
                    )

                _log_active_tab(f"Found {len(records)} channels via QPTIFF metadata.")
                return _make_records_unique(records), mpp

            series = _get_primary_series(tif)
            if series is not None:
                fallback_shape = getattr(series, 'shape', ())
                fallback_axes = getattr(series, 'axes', '')
            elif tif.pages:
                fallback_shape = tif.pages[0].shape
                fallback_axes = getattr(tif.pages[0], 'axes', '')
            else:
                fallback_shape = ()
                fallback_axes = ''

            generic_records = _build_generic_channel_records(fallback_shape, fallback_axes)
            if generic_records:
                _log_active_tab(
                    f"Found {len(generic_records)} channels via generic TIFF fallback "
                    f"(shape={tuple(fallback_shape)}, axes={fallback_axes or 'unknown'})."
                )
                return _make_records_unique(generic_records), mpp

            raise ValueError("Could not extract channel info from OME-XML or QPTIFF metadata.")

    except Exception as e:
        _log_active_tab(f"Error reading channel info: {e}")
        return [], mpp


def get_channel_info(image_path):
    """
    Extracts channel names, colors, and other metadata from an OME-TIFF or QPTIFF file.
    """
    channel_records, mpp = get_channel_records(image_path)
    channel_names = [record['name'] for record in channel_records]
    channel_colors = [record['color'] for record in channel_records]
    channel_metadata = [record['metadata'] for record in channel_records]
    return channel_names, channel_colors, channel_metadata, mpp


def subtract_background_and_save(
    input_path, 
    output_path, 
    blank_mapping, 
    channel_names, 
    channel_colors, 
    mpp
):
    """
    Performs background subtraction and saves the result as a pyramidal OME-TIFF.
    """
    log_time("Starting background subtraction process...")
    
    # Invert mapping for easier lookup: marker -> blank
    marker_to_blank = {}
    for blank, markers in blank_mapping.items():
        for marker in markers:
            marker_to_blank[marker] = blank
    
    # Build output channel lists (all channels EXCEPT blank channels themselves)
    # Preserve the original order and colors
    output_channel_names = []
    output_channel_colors = []
    
    for i, name in enumerate(channel_names):
        if name not in blank_mapping:  # Include all channels except blanks
            output_channel_names.append(name)
            output_channel_colors.append(channel_colors[i])
    
    # Identify channels to keep as-is
    no_subtraction_channels = [
        name for name in channel_names 
        if name not in marker_to_blank and name not in blank_mapping
    ]
    log_time(f"Found {len(marker_to_blank)} markers for subtraction.")
    log_time(f"Found {len(no_subtraction_channels)} channels to keep as-is.")
    log_time(f"Output will have {len(output_channel_names)} channels (blank channels excluded).")

    try:
        with tifffile.TiffFile(input_path) as tif:
            series = _get_primary_series(tif)
            if series is not None:
                img_height, img_width = _get_spatial_shape(series.shape, getattr(series, 'axes', None))
            else:
                base_page = tif.pages[0]
                img_height, img_width = _get_spatial_shape(base_page.shape, getattr(base_page, 'axes', None))
            img_dtype = tif.pages[0].dtype

            # Create new OME-XML for the output file (only includes non-blank channels)
            ome_xml = create_ome_metadata(
                channel_names=output_channel_names,
                channel_colors=output_channel_colors,
                SizeX=img_width,
                SizeY=img_height,
                pixel_type=str(img_dtype),
                n_channels=len(output_channel_names),
                physical_size_x=mpp['codex'],
                physical_size_y=mpp['codex']
            )

            # Prepare to write output
            tile_size = (256, 256)
            n_pyramid_levels = 3
            
            with tifffile.TiffWriter(output_path, bigtiff=True) as tiff_writer:
                # Load blank channels into memory
                blank_data = {}
                for blank_name in blank_mapping.keys():
                    if blank_name in channel_names:
                        idx = channel_names.index(blank_name)
                        log_time(f"Loading blank channel: {blank_name} (index {idx})")
                        blank_data[blank_name] = _read_channel_image_from_tif(tif, idx).astype(np.int32)
                    else:
                        log_time(f"Warning: Blank channel '{blank_name}' not found in image.")

                # Process and write each channel
                output_channel_idx = 0  # Track which output channel we're writing
                for i, ch_name in enumerate(channel_names):
                    log_time(f"Processing channel {i+1}/{len(channel_names)}: {ch_name}")
                    
                    # Skip the original blank channels themselves
                    if ch_name in blank_mapping:
                        log_time(f"  -> Skipping original blank channel.")
                        continue

                    marker_img = _read_channel_image_from_tif(tif, i)
                    
                    # Perform subtraction if needed
                    if ch_name in marker_to_blank:
                        blank_name = marker_to_blank[ch_name]
                        if blank_name in blank_data:
                            log_time(f"  -> Subtracting background from '{blank_name}'")
                            blank_img = blank_data[blank_name]
                            # Ensure shapes match before subtraction
                            if marker_img.shape == blank_img.shape:
                                subtracted_img = np.maximum(0, marker_img.astype(np.int32) - blank_img.astype(np.int32))
                                final_img = subtracted_img.astype(img_dtype)
                            else:
                                log_time(f"  -> Shape mismatch between {ch_name} and {blank_name}. Skipping subtraction.")
                                final_img = marker_img
                        else:
                            log_time(f"  -> Blank '{blank_name}' not loaded. Keeping original.")
                            final_img = marker_img
                    else:
                        log_time(f"  -> No subtraction needed. Keeping original.")
                        final_img = marker_img

                    # Write base resolution (write OME-XML only for the first output channel)
                    tiff_writer.write(
                        final_img,
                        subifds=n_pyramid_levels,
                        photometric='minisblack',
                        compression='deflate',
                        tile=tile_size,
                        description=ome_xml if output_channel_idx == 0 else None,
                        planarconfig='CONTIG',
                    )
                    
                    # Create and write pyramid levels
                    prev_img = final_img
                    for level in range(n_pyramid_levels):
                        scale = 2 ** (level + 1)
                        new_width = max(1, img_width // scale)
                        new_height = max(1, img_height // scale)
                        downsampled_img = cv2.resize(prev_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        tiff_writer.write(
                            downsampled_img,
                            photometric='minisblack',
                            compression='deflate',
                            tile=tile_size,
                            planarconfig='CONTIG',
                        )
                    
                    output_channel_idx += 1  # Increment output channel counter
                    
        log_time(f"Successfully saved background-subtracted image to: {output_path}")

    except Exception as e:
        import traceback
        log_time(f"An error occurred during background subtraction: {e}")
        log_time(traceback.format_exc())

