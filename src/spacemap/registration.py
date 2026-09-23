"""
Registration functions for SPACEMAP_V4 pipeline.

This module contains functions for image registration, merging, and ROI extraction
used in the SPACEMAP pipeline.
"""

import os
import cv2
import zlib
import tifffile
import numpy as np
import matplotlib.pyplot as plt
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

import palom
import dask.array as da
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from spacemap.utils import (
    log_time, check_file_exists, create_ome_metadata, rgb_to_color_int,
    load_geojson, extract_rois_with_geometry_from_geojson
)
from spacemap.background_subtraction_utils import (
    get_channel_info, read_channel_image, read_channel_from_open_tif, normalize_image_stem, is_supported_image_file, io_worker_count,
)

try:
    import imagecodecs
except ImportError:  # imagecodecs is a tifffile dependency, but degrade rather than fail
    imagecodecs = None


# Tile sizes for the TIFFs this module writes.
#
# Tiling is what makes partial reads cheap: a strip-based TIFF has to decode the full
# image width to serve any crop, which is why ROI extraction (tab 3) reads the merged
# stack tile by tile.  The registration reference uses 1024 because palom turns TIFF
# tiles into dask chunks and derives its block-shift grid from them — 1024 reproduces
# the 53x21 grid palom used when it had to rechunk the untiled reference itself.
_OUTPUT_TILE = (512, 512)
_MOVING_TILE = (512, 512)
_REFERENCE_TILE = (1024, 1024)


REFERENCE_CHANNEL_TERMS = (
    'dapi',
    'hoechst',
    'hoescht',
    'nuclear',
    'nuclei',
)


def _sanitize_reference_label(reference_value):
    """Create a stable filesystem-safe label for a reference channel override."""
    text = str(reference_value or '').strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text or 'reference'


def _find_reference_channel(channel_names):
    """Return the best nuclear-reference channel index and name."""
    if not channel_names:
        return 0, None

    for idx, name in enumerate(channel_names):
        if not isinstance(name, str):
            continue
        lowered = name.lower()
        if any(term in lowered for term in REFERENCE_CHANNEL_TERMS):
            return idx, name

    return 0, channel_names[0]


def resolve_reference_channel(channel_names, reference_channel=None, reference_channel_index=None):
    """Resolve a manual or auto-detected registration reference channel."""
    if not channel_names:
        return 0, None

    if reference_channel_index is not None:
        try:
            resolved_index = int(reference_channel_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference channel index: {reference_channel_index}") from exc

        if resolved_index < 0 or resolved_index >= len(channel_names):
            raise ValueError(
                f"Reference channel index {resolved_index} is out of range for {len(channel_names)} channels"
            )
        return resolved_index, channel_names[resolved_index]

    if reference_channel:
        desired = str(reference_channel).strip().lower()
        exact_matches = [
            (idx, name) for idx, name in enumerate(channel_names)
            if str(name).strip().lower() == desired
        ]
        if exact_matches:
            return exact_matches[0]

        partial_matches = [
            (idx, name) for idx, name in enumerate(channel_names)
            if desired in str(name).strip().lower()
        ]
        if partial_matches:
            return partial_matches[0]

        raise ValueError(
            f"Reference channel '{reference_channel}' was not found. "
            f"Available channels: {list(channel_names)}"
        )

    return _find_reference_channel(channel_names)


def resolve_reference_channel_from_image(image_path, reference_channel=None, reference_channel_index=None):
    """Resolve the reference channel from image metadata, honoring extracted single-channel files."""
    channel_names, _channel_colors, _channel_meta, _mpp = get_channel_info(image_path)
    if len(channel_names) <= 1:
        return 0, channel_names[0] if channel_names else None
    return resolve_reference_channel(channel_names, reference_channel, reference_channel_index)


def _get_primary_series(tif):
    """Return the series most likely to represent the full-resolution image."""
    if not tif.series:
        return None
    return max(tif.series, key=lambda series: np.prod(getattr(series, 'shape', (0, 0))))


def _image_base_name(image_path, suffix_to_strip=None):
    """Return a normalized image stem across qptiff and OME-TIFF variants."""
    name = normalize_image_stem(image_path)

    if suffix_to_strip and name.endswith(suffix_to_strip):
        name = name[:-len(suffix_to_strip)]

    return name


def _is_pyramidal_multichannel_ome(image_path):
    """Detect OME-TIFF files that should be consumed lazily instead of materialized."""
    try:
        with tifffile.TiffFile(image_path) as tif:
            if not tif.ome_metadata:
                return False
            series = _get_primary_series(tif)
            if series is None:
                return False
            axes = getattr(series, 'axes', '') or ''
            level_count = len(getattr(series, 'levels', []))
            shape = getattr(series, 'shape', ())
            has_channel_axis = any(axis in axes for axis in ('C', 'I', 'S'))
            return level_count > 1 and has_channel_axis and len(shape) >= 3
    except Exception:
        return False


def _ome_channel_ifd_map(ome_xml):
    """Map OME channel indices to TIFF IFDs when TiffData entries are present."""
    if not ome_xml:
        return {}

    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    root = ET.fromstring(ome_xml)
    pixels = root.find('.//ome:Pixels', namespace)
    if pixels is None:
        pixels = root.find('.//Pixels')
    if pixels is None:
        return {}

    channel_ifd_map = {}
    for tiff_data in pixels.findall('ome:TiffData', namespace) or pixels.findall('TiffData'):
        try:
            first_c = int(tiff_data.attrib.get('FirstC', '0'))
            ifd = int(tiff_data.attrib['IFD'])
        except (KeyError, TypeError, ValueError):
            continue
        channel_ifd_map.setdefault(first_c, ifd)

    return channel_ifd_map


def _read_ome_channel_plane(tif, channel_index):
    """Read a single full-resolution channel plane from an OME-TIFF."""
    channel_ifd_map = _ome_channel_ifd_map(tif.ome_metadata)
    if channel_index in channel_ifd_map and channel_ifd_map[channel_index] < len(tif.pages):
        page = tif.pages[channel_ifd_map[channel_index]]
        if getattr(page, 'ndim', 0) == 2:
            return page.asarray()

    if channel_index < len(tif.pages):
        page = tif.pages[channel_index]
        if getattr(page, 'ndim', 0) == 2:
            return page.asarray()

    series = _get_primary_series(tif)
    if series is None:
        raise ValueError('No readable TIFF series found')

    axes = getattr(series, 'axes', '') or ''
    zarr_store = series.aszarr(level=0)
    import zarr

    zarr_array = zarr.open(zarr_store, mode='r')
    if 'C' in axes:
        channel_axis = axes.index('C')
    elif 'I' in axes:
        channel_axis = axes.index('I')
    elif 'S' in axes:
        channel_axis = axes.index('S')
    else:
        raise ValueError(f'Unable to identify channel axis from axes={axes!r}')

    selector = [0] * zarr_array.ndim
    for axis_idx, axis_name in enumerate(axes):
        if axis_name in ('Y', 'X'):
            selector[axis_idx] = slice(None)
        elif axis_idx == channel_axis:
            selector[axis_idx] = channel_index
    return np.asarray(zarr_array[tuple(selector)])


def _normalize_image_axes(image_array, axes=''):
    """Collapse non-spatial axes and return image data as YX or YXC."""
    image = np.asarray(image_array)
    axes = str(axes or '')

    if image.ndim == 0:
        raise ValueError('Image array is scalar and cannot be resized')

    if axes and len(axes) == image.ndim and 'Y' in axes and 'X' in axes:
        selector = [slice(None)] * image.ndim
        channel_axis = None
        for axis_index, axis_name in enumerate(axes):
            if axis_name in ('Y', 'X'):
                continue
            if axis_name in ('C', 'S', 'I') and channel_axis is None:
                channel_axis = axis_index
            else:
                selector[axis_index] = 0

        image = np.asarray(image[tuple(selector)])
        reduced_axes = ''.join(axis_name for axis_name, selected in zip(axes, selector) if isinstance(selected, slice))

        if image.ndim == 3 and channel_axis is not None:
            current_channel_axis = reduced_axes.index(next(axis for axis in reduced_axes if axis in ('C', 'S', 'I')))
            y_axis = reduced_axes.index('Y')
            x_axis = reduced_axes.index('X')
            image = np.moveaxis(image, [y_axis, x_axis, current_channel_axis], [0, 1, 2])
        elif image.ndim == 2 and set(reduced_axes) >= {'Y', 'X'}:
            y_axis = reduced_axes.index('Y')
            x_axis = reduced_axes.index('X')
            image = np.moveaxis(image, [y_axis, x_axis], [0, 1])

    image = np.squeeze(image)

    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            return image
        if image.shape[0] in (3, 4):
            return np.moveaxis(image, 0, -1)
        return image[..., :1]

    raise ValueError(f'Unsupported H&E image shape after normalization: {image.shape}')


def _load_he_image_for_preprocessing(he_path, target_scale_factor=1.0):
    """Load H&E image data as YX or YXC using the best source pyramid level for the requested scale.

    Level shapes come from TIFF metadata and only the selected level is decoded, so a
    multi-gigapixel slide is never materialized just to choose between levels.
    """
    with tifffile.TiffFile(he_path) as tif:
        series = _get_primary_series(tif)
        if series is None:
            return _normalize_image_axes(tif.asarray(maxworkers=io_worker_count()), ''), 1.0

        series_axes = getattr(series, 'axes', '') or ''
        levels = list(getattr(series, 'levels', [])) or [series]
        best_level = levels[0]
        best_level_scale = 1.0

        if target_scale_factor < 1.0 and len(levels) > 1:
            full_height = max(1, _level_spatial_shape(levels[0], series_axes)[0])
            level_scales = [
                float(_level_spatial_shape(level, series_axes)[0]) / float(full_height)
                for level in levels
            ]
            best_index = min(
                range(len(levels)),
                key=lambda index: abs(level_scales[index] - target_scale_factor),
            )
            best_level = levels[best_index]
            best_level_scale = level_scales[best_index]
            log_time(
                f"Selected H&E source level {best_index} of {len(levels)} "
                f"(scale {best_level_scale:.4f}, target {target_scale_factor:.4f})"
            )

        level_axes = getattr(best_level, 'axes', '') or series_axes
        return (
            _normalize_image_axes(best_level.asarray(maxworkers=io_worker_count()), level_axes),
            best_level_scale,
        )


def _level_spatial_shape(level, fallback_axes=''):
    """Spatial (height, width) of a pyramid level, read from metadata without decoding."""
    axes = getattr(level, 'axes', '') or fallback_axes or ''
    shape = tuple(int(dim) for dim in (getattr(level, 'shape', ()) or ()))
    if axes and 'Y' in axes and 'X' in axes and len(axes) == len(shape):
        return shape[axes.index('Y')], shape[axes.index('X')]
    if len(shape) >= 3 and shape[-1] in (3, 4):
        return shape[-3], shape[-2]
    if len(shape) >= 2:
        return shape[-2], shape[-1]
    raise ValueError(f'Cannot determine spatial shape from level shape {shape} (axes={axes!r})')


def _iter_plane_tiles(plane, tile):
    """Split one plane into the zero-padded tiles ``TiffWriter`` expects.

    When ``data`` is an iterator and ``tile`` is set, tifffile consumes tiles rather than
    planes, in row-major order per plane, each of shape ``tile + (samples,)``.
    """
    tile_height, tile_width = tile
    height, width = plane.shape[:2]
    chunk_shape = (tile_height, tile_width, 1)
    for y0 in range(0, height, tile_height):
        rows = min(tile_height, height - y0)
        for x0 in range(0, width, tile_width):
            cols = min(tile_width, width - x0)
            chunk = plane[y0:y0 + rows, x0:x0 + cols, np.newaxis]
            if rows != tile_height or cols != tile_width:
                padded = np.zeros(chunk_shape, plane.dtype)
                padded[:rows, :cols] = chunk
                chunk = padded
            yield chunk


def _write_tiled_pyramid(output_path, image, photometric, axes, n_pyramid_levels=4, tile=_MOVING_TILE):
    """Write a single image as a tiled, pyramidal TIFF.

    palom reads registration inputs through a dask view of the TIFF. Tiles become dask
    chunks, and stored pyramid levels are used directly for the coarse-alignment
    thumbnails; without them palom rebuilds every level with ``da.coarsen`` over the
    full-resolution array, which dominates the registration step on whole-slide inputs.
    """
    base = np.ascontiguousarray(image)
    workers = io_worker_count()

    # tifffile only exposes the subifd pyramid as ``series.levels`` when an RGB third
    # axis is read as samples. Labelling it 'YXC' turns it into a channel axis and the
    # levels silently disappear, which puts palom back on the slow da.coarsen path.
    if base.ndim == 3 and axes.upper() in ('YXC', 'YXS'):
        axes = 'YXS'

    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        writer.write(
            base,
            photometric=photometric,
            compression=('deflate', 1),
            tile=tile,
            subifds=n_pyramid_levels,
            metadata={'axes': axes},
            maxworkers=workers,
        )
        level = base
        for _ in range(n_pyramid_levels):
            level = cv2.resize(
                level,
                (max(1, level.shape[1] // 2), max(1, level.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            writer.write(
                np.ascontiguousarray(level),
                photometric=photometric,
                compression=('deflate', 1),
                tile=tile,
                subfiletype=1,
                maxworkers=workers,
            )


MIN_RELIABLE_KEYPOINTS = 2000
_MAX_COARSE_SCALE = 4.0


def _validate_coarse_affine(affine_matrix, n_keypoints):
    """Reject a coarse affine that block warping cannot survive.

    Too few keypoints leaves the affine under-determined. palom then inverts it per
    block to decide which part of the moving image to read, and a wild transform asks
    for a source region beyond OpenCV's 32767 px ``remap`` limit — surfacing as a bare
    assertion failure deep inside ``cv2.warpAffine`` with no hint of the real cause.
    """
    matrix = np.asarray(affine_matrix, dtype=float)
    linear = matrix[:2, :2]

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Coarse alignment produced a non-finite transform using {n_keypoints} keypoints. "
            f"Raise 'Alignment Keypoints' (>= {MIN_RELIABLE_KEYPOINTS}) and confirm the two "
            "images overlap and any pre-rotation is correct."
        )

    scales = np.linalg.svd(linear, compute_uv=False)
    if scales.min() <= 1.0 / _MAX_COARSE_SCALE or scales.max() >= _MAX_COARSE_SCALE:
        raise ValueError(
            f"Coarse alignment produced an implausible transform (scale factors "
            f"{scales.min():.3g}-{scales.max():.3g}) using {n_keypoints} keypoints. "
            f"Raise 'Alignment Keypoints' (>= {MIN_RELIABLE_KEYPOINTS}) and confirm the two "
            "images overlap and any pre-rotation is correct."
        )

    if n_keypoints < MIN_RELIABLE_KEYPOINTS:
        log_time(
            f"Warning: coarse alignment used only {n_keypoints} keypoints; "
            f"{MIN_RELIABLE_KEYPOINTS}+ is recommended for whole-slide images."
        )


def _pack_cached_plane(plane):
    """Compress a reduced plane for the merge pyramid cache. Round-trips byte-exactly."""
    raw = plane.tobytes()
    if imagecodecs is not None:
        try:
            return 'zstd', imagecodecs.zstd_encode(raw, 1), len(raw)
        except Exception:
            pass
    return 'zlib', zlib.compress(raw, 1), len(raw)


def _unpack_cached_plane(codec, payload, nbytes, shape, dtype):
    if codec == 'zstd':
        raw = imagecodecs.zstd_decode(payload)
    else:
        raw = zlib.decompress(payload)
    return np.frombuffer(raw, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize).reshape(shape)


def _write_pyramidal_plane(
    tiff_writer,
    image,
    full_width,
    full_height,
    n_pyramid_levels=3,
    tile_size=(256, 256),
    description=None,
):
    """Write one base-resolution plane followed by reduced pyramid levels."""
    base_image = np.ascontiguousarray(image)
    tiff_writer.write(
        base_image,
        subifds=n_pyramid_levels,
        photometric='minisblack',
        compression='deflate',
        tile=tile_size,
        description=description,
    )

    prev_img = base_image
    for lvl in range(n_pyramid_levels):
        scale = 2 ** (lvl + 1)
        new_width = max(1, full_width // scale)
        new_height = max(1, full_height // scale)
        downsampled_img = cv2.resize(prev_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        tiff_writer.write(
            np.ascontiguousarray(downsampled_img),
            photometric='minisblack',
            compression='deflate',
            tile=tile_size,
            subfiletype=1,
        )
        prev_img = downsampled_img


def _coerce_uint16_plane(image):
    plane = np.asarray(image)
    if plane.dtype == np.uint8:
        plane = plane.astype(np.uint16) * 257
    elif plane.dtype == np.float32 or plane.dtype == np.float64:
        if plane.max() <= 1.0:
            plane = (plane * 65535.0).clip(0, 65535).astype(np.uint16)
        else:
            plane = plane.clip(0, 65535).astype(np.uint16)
    elif plane.dtype != np.uint16:
        plane = plane.astype(np.uint16)
    return np.ascontiguousarray(plane)


_OPENCV_MAX_DIM = 32766  # cv2.warpAffine hard limit (SHRT_MAX - 1)


def _rotate_image(image, angle_deg):
    """Rotate image counter-clockwise by angle_deg degrees, expanding canvas to fit full content.

    Special cases:
    - Multiples of 90° use ``np.rot90`` (exact, lossless, no size limit).
    - For arbitrary angles on images where the expanded canvas would exceed
      OpenCV's SHRT_MAX limit, the rotation is applied without canvas expansion
      (corners are clipped).  This is acceptable because this image is used only
      for keypoint matching, not as a final output.
    """
    if abs(angle_deg) < 0.01:
        return image

    # For multiples of 90°: use numpy — exact, no interpolation, no size limit.
    nearest_quarters = round(angle_deg / 90.0)
    if abs(angle_deg - nearest_quarters * 90.0) < 0.5:
        k = int(nearest_quarters) % 4
        return np.rot90(image, k=k)

    # Arbitrary angle: compute expanded canvas.
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))

    if new_w >= _OPENCV_MAX_DIM or new_h >= _OPENCV_MAX_DIM:
        # Image is too large for an expanded canvas; rotate in-place (clips corners).
        # Acceptable for pre-processing before keypoint-based alignment.
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def _rot90_inv_matrix(k: int, H_orig: int, W_orig: int) -> np.ndarray:
    """3×3 homogeneous matrix that maps np.rot90(img, k=k) pixel coords back to original coords.

    All four 90° multiples are handled exactly; k is reduced mod 4.
    """
    k = int(k) % 4
    if k == 0:
        return np.eye(3, dtype=float)
    elif k == 1:   # 90° CCW  → inverse: x_orig = W-1-y_rot,  y_orig = x_rot
        return np.array([[0, -1, W_orig - 1], [1, 0, 0], [0, 0, 1]], dtype=float)
    elif k == 2:   # 180°     → inverse: x_orig = W-1-x_rot,  y_orig = H-1-y_rot
        return np.array([[-1, 0, W_orig - 1], [0, -1, H_orig - 1], [0, 0, 1]], dtype=float)
    else:          # 270° CCW → inverse: x_orig = y_rot,       y_orig = H-1-x_rot
        return np.array([[0, 1, 0], [-1, 0, H_orig - 1], [0, 0, 1]], dtype=float)


def _spatial_shape_from_image(image_path):
    with tifffile.TiffFile(image_path) as tif:
        series = _get_primary_series(tif)
        if series is not None:
            axes = getattr(series, 'axes', '') or ''
            shape = getattr(series, 'shape', ())
            if 'Y' in axes and 'X' in axes:
                return (int(shape[axes.index('Y')]), int(shape[axes.index('X')]))

        if tif.pages:
            page_shape = tif.pages[0].shape
            return (int(page_shape[0]), int(page_shape[1]))

    raise ValueError(f"Could not determine spatial shape for {image_path}")


def _build_ome_metadata_dict(channel_names, channel_colors, pixel_size, channel_metadata=None):
    channel_section = {
        'Name': [str(name) for name in channel_names],
        'Color': [rgb_to_color_int(color) for color in channel_colors],
    }

    if channel_metadata:
        fluor_values = []
        has_fluor = False
        for meta in channel_metadata:
            fluor = str((meta or {}).get('dye', '') or '').strip()
            if fluor and fluor != '-':
                has_fluor = True
            fluor_values.append(fluor)
        if has_fluor:
            channel_section['Fluor'] = fluor_values

    return {
        'axes': 'CYX',
        'PhysicalSizeX': float(pixel_size),
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': float(pixel_size),
        'PhysicalSizeYUnit': 'µm',
        'Channel': channel_section,
    }


def _write_ome_pyramid_from_plane_factory(
    output_path,
    plane_factory,
    channel_names,
    channel_colors,
    pixel_size,
    base_shape,
    channel_metadata=None,
    n_pyramid_levels=3,
    tile=_OUTPUT_TILE,
):
    base_height, base_width = map(int, base_shape)
    metadata = _build_ome_metadata_dict(
        channel_names=channel_names,
        channel_colors=channel_colors,
        pixel_size=pixel_size,
        channel_metadata=channel_metadata,
    )
    workers = io_worker_count()

    def tiled_chunks(level, level_shape):
        for plane in plane_factory(level, level_shape):
            yield from _iter_plane_tiles(plane, tile)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff_writer:
        tiff_writer.write(
            data=tiled_chunks(0, (base_height, base_width)),
            shape=(len(channel_names), base_height, base_width),
            dtype=np.uint16,
            photometric='minisblack',
            compression=('deflate', 1),
            tile=tile,
            subifds=n_pyramid_levels,
            metadata=metadata,
            maxworkers=workers,
        )

        for level in range(1, n_pyramid_levels + 1):
            level_shape = (
                max(1, base_height // (2 ** level)),
                max(1, base_width // (2 ** level)),
            )
            tiff_writer.write(
                data=tiled_chunks(level, level_shape),
                shape=(len(channel_names), level_shape[0], level_shape[1]),
                dtype=np.uint16,
                photometric='minisblack',
                compression=('deflate', 1),
                tile=tile,
                subfiletype=1,
                maxworkers=workers,
            )


def preprocess_he_image(he_path, output_dir, mpp, overwrite=False, pre_rotation_deg=0.0):
    """Preprocess H&E image - resize to match CODEX resolution, optionally pre-rotating."""

    base_name = _image_base_name(he_path, suffix_to_strip='_he')
    rot_suffix = f"_rot{int(round(pre_rotation_deg))}" if abs(pre_rotation_deg) > 0.01 else ""
    output_path = os.path.join(output_dir, f"{base_name}_he_resized{rot_suffix}.tiff")
    
    if check_file_exists(output_path, "H&E preprocessing", overwrite=overwrite):
        return output_path
    
    log_time(f"Preprocessing H&E image: {base_name}")
    scale_factor = mpp['hne'] / mpp['codex']

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(f"Invalid H&E resize scale factor {scale_factor} from MPP values {mpp}")

    if abs(scale_factor - 1.0) < 1e-6 and _is_pyramidal_multichannel_ome(he_path):
        log_time("H&E image is already a pyramidal OME-TIFF at the target resolution; reusing source image.")
        return he_path
    
    he_img, source_scale = _load_he_image_for_preprocessing(he_path, target_scale_factor=scale_factor)
    log_time(f"H&E image loaded: {he_img.shape} (source scale {source_scale:.4f})")
    
    # Resize to match CODEX resolution
    spatial_height, spatial_width = he_img.shape[:2]
    residual_scale = scale_factor / max(source_scale, 1e-12)
    if abs(residual_scale - 1.0) < 1e-6:
        he_resized = he_img
    else:
        new_height = max(1, int(round(spatial_height * residual_scale)))
        new_width = max(1, int(round(spatial_width * residual_scale)))
        interpolation = cv2.INTER_AREA if residual_scale < 1.0 else cv2.INTER_LINEAR
        he_resized = cv2.resize(he_img, (new_width, new_height), interpolation=interpolation)

    # Apply pre-rotation if requested (facilitates palom alignment)
    if abs(pre_rotation_deg) > 0.01:
        log_time(f"Applying pre-rotation of {pre_rotation_deg:.1f}° to H&E image...")
        he_resized = _rotate_image(he_resized, pre_rotation_deg)
        log_time(f"H&E rotated shape: {he_resized.shape}")

    axes = 'YXC' if he_resized.ndim == 3 else 'YX'
    photometric = 'rgb' if he_resized.ndim == 3 and he_resized.shape[-1] in (3, 4) else 'minisblack'

    _write_tiled_pyramid(output_path, he_resized, photometric=photometric, axes=axes, tile=_MOVING_TILE)

    log_time(f"H&E resized and saved: {output_path}")
    return output_path


def extract_dapi_channel(codex_path, output_dir, overwrite=False, reference_channel=None, reference_channel_index=None):
    """Extract the registration reference channel from a CODEX image."""
    
    base_name = _image_base_name(codex_path, suffix_to_strip='_codex')
    if reference_channel is not None:
        output_suffix = f"_reference_{_sanitize_reference_label(reference_channel)}.tiff"
    elif reference_channel_index is not None:
        output_suffix = f"_reference_idx{int(reference_channel_index)}.tiff"
    else:
        output_suffix = "_dapi.tiff"
    output_path = os.path.join(output_dir, f"{base_name}{output_suffix}")
    
    if check_file_exists(output_path, "DAPI extraction", overwrite=overwrite):
        return output_path
    
    log_time(f"Extracting registration reference channel: {base_name}")

    channel_names, _channel_colors, _channel_meta, _mpp = get_channel_info(codex_path)
    ref_idx, ref_name = resolve_reference_channel(channel_names, reference_channel, reference_channel_index)
    if ref_name is None:
        log_time("Reference channel not found by metadata; using first channel as fallback")
    else:
        log_time(f"Using reference channel at index {ref_idx}: {ref_name}")

    if _is_pyramidal_multichannel_ome(codex_path):
        log_time("Detected pyramidal multi-channel OME-TIFF; using source image directly for registration reference.")
        return codex_path

    with tifffile.TiffFile(codex_path) as tif:
        if tif.ome_metadata:
            dapi_img = _read_ome_channel_plane(tif, ref_idx)
        else:
            codex_img = tif.asarray()
            log_time(f"CODEX image loaded: {codex_img.shape}")

            if codex_img.ndim == 3:
                # Decide which layout the 3D array has.
                # Standard layouts: (C, H, W) or (H, W, C)
                is_ch_first = False
                if channel_names:
                    if codex_img.shape[0] == len(channel_names):
                        is_ch_first = True
                    elif codex_img.shape[2] == len(channel_names):
                        is_ch_first = False
                    elif abs(codex_img.shape[0] - len(channel_names)) < abs(codex_img.shape[2] - len(channel_names)):
                        is_ch_first = True
                    else:
                        is_ch_first = False
                else:
                    is_ch_first = codex_img.shape[0] < codex_img.shape[2]

                if is_ch_first:
                    log_time(f"CODEX format detected: (channels={codex_img.shape[0]}, height={codex_img.shape[1]}, width={codex_img.shape[2]})")
                    safe_ref_idx = max(0, min(ref_idx, codex_img.shape[0] - 1))
                    dapi_img = codex_img[safe_ref_idx, :, :]
                else:
                    log_time(f"CODEX format detected: (height={codex_img.shape[0]}, width={codex_img.shape[1]}, channels={codex_img.shape[2]})")
                    safe_ref_idx = max(0, min(ref_idx, codex_img.shape[2] - 1))
                    dapi_img = codex_img[:, :, safe_ref_idx]
            elif codex_img.ndim == 2:
                dapi_img = codex_img
            else:
                log_time(f"Warning: Unusual CODEX dimensions {codex_img.shape}")
                safe_ref_idx = max(0, min(ref_idx, codex_img.shape[0] - 1))
                dapi_img = codex_img[safe_ref_idx] if codex_img.shape[0] > safe_ref_idx else codex_img[0]
    
    # Ensure 2D output
    if dapi_img.ndim != 2:
        log_time(f"Error: DAPI extraction resulted in {dapi_img.ndim}D array, expected 2D")
        # Force to 2D by taking the first valid 2D slice
        while dapi_img.ndim > 2:
            dapi_img = dapi_img[0]
    
    # Ensure dimensions are compatible with palom pyramid creation
    h, w = dapi_img.shape
    log_time(f"Reference dimensions: {h} x {w}")
    
    # Check if dimensions will cause issues with pyramid downsampling
    min_dim = min(h, w)
    if min_dim < 64:
        log_time(f"Warning: Small dimension detected ({min_dim}), may cause registration issues")
    
    # Save as a tiled, pyramidal TIFF so palom reads it as 1024x1024 dask chunks with
    # real thumbnail levels instead of one whole-image chunk it has to rechunk itself.
    _write_tiled_pyramid(output_path, dapi_img, photometric='minisblack', axes='YX', tile=_REFERENCE_TILE)

    log_time(f"Reference channel saved: {output_path}")
    return output_path


def register_images(
    he_resized_path,
    dapi_path,
    output_dir,
    mpp,
    overwrite=False,
    reference_channel=None,
    reference_channel_index=None,
    n_keypoints=10000,
):
    """Register H&E to a CODEX reference channel with error handling."""
    base_name = _image_base_name(he_resized_path)
    if base_name.endswith('_he_resized'):
        base_name = base_name[:-len('_he_resized')]
    elif base_name.endswith('_he'):
        base_name = base_name[:-len('_he')]
    output_path = os.path.join(output_dir, f"{base_name}_he_registered.ome.tiff")
    ref_channel_index = 0

    manual_override = reference_channel is not None or reference_channel_index is not None
    try:
        ref_channel_index, ref_name = resolve_reference_channel_from_image(
            dapi_path,
            reference_channel=reference_channel,
            reference_channel_index=reference_channel_index,
        )
        if ref_name is not None:
            log_time(f"Resolved registration reference channel: {ref_name} (index {ref_channel_index})")
    except Exception as exc:
        if manual_override:
            raise
        log_time(f"Could not resolve registration reference channel from metadata: {exc}")
    
    try:
        register_he_to_codex(
            dapi_path,
            he_resized_path,
            output_path,
            mpp,
            overwrite=overwrite,
            ref_channel_index=ref_channel_index,
            n_keypoints=n_keypoints,
        )
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"Registration reported success but produced no output: {output_path}"
            )
        return output_path
    except Exception as e:
        # Deliberately not substituting the unregistered H&E here. Doing so used to let
        # the run continue and report success while the merged stack carried H&E channels
        # that were never aligned to the reference — a silent scientific error. Fail the
        # pair instead so the caller records the real reason.
        import traceback
        log_time(f"Registration failed for {base_name}: {e}\n{traceback.format_exc()}")
        raise


def merge_channels(registered_he_path, codex_path, output_dir, base_name, mpp, channel_colors_override=None, channels_to_keep=None, overwrite=False):
    """Merge registered H&E with CODEX channels"""
    output_path = os.path.join(output_dir, f"{base_name}_merged.ome.tiff")
    
    save_merged_pyramidal_ome_tiff(codex_path, registered_he_path, output_path, mpp, channel_colors_override, channels_to_keep, overwrite=overwrite)
    return output_path


def register_he_to_codex(codex_dapi_path, he_img_path, registered_he_output, mpp, overwrite=False, ref_channel_index=0, n_keypoints=10000):
    """
    Register the H&E image (grayscale) to the CODEX DAPI channel.
    Uses palom's affine registration (with a coarse registration using ~10,000 keypoints).
    Saves the registered H&E image as an OME-TIFF.
    
    Args:
        codex_dapi_path (str): Path to CODEX DAPI channel image
        he_img_path (str): Path to H&E image
        registered_he_output (str): Output path for registered H&E image
        mpp (dict): Dictionary containing pixel sizes for 'codex' and 'hne'
        overwrite (bool): Force overwrite if file exists
    """
    if check_file_exists(registered_he_output, "H&E registration", overwrite=overwrite):
        return

    log_time("Starting registration process...")
    try:
        THUMBNAIL_LEVEL = 2
        LEVEL = 0

        ref_reader = palom.reader.OmePyramidReader(codex_dapi_path)
        ref_reader._pixel_size = mpp['codex']

        mov_reader = palom.reader.OmePyramidReader(he_img_path)

        ref_thumb = ref_reader.read_level_channels(THUMBNAIL_LEVEL, ref_channel_index).compute()
        mov_thumb = mov_reader.read_level_channels(THUMBNAIL_LEVEL, 0).compute()
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(ref_thumb, cmap='gray')
        plt.title("CODEX Reference Thumbnail")
        plt.subplot(1,2,2)
        plt.imshow(mov_thumb, cmap='gray')
        plt.title("H&E Thumbnail")
        plt.show()

        aligner = palom.align.Aligner(
            ref_img=ref_reader.read_level_channels(LEVEL, ref_channel_index),
            moving_img=mov_reader.read_level_channels(LEVEL, 0),
            ref_thumbnail=ref_thumb,
            moving_thumbnail=mov_thumb,
            ref_thumbnail_down_factor=ref_reader.level_downsamples[THUMBNAIL_LEVEL] / ref_reader.level_downsamples[LEVEL],
            moving_thumbnail_down_factor=mov_reader.level_downsamples[THUMBNAIL_LEVEL] / mov_reader.level_downsamples[LEVEL]
        )
        aligner.coarse_register_affine(n_keypoints=int(n_keypoints))
        _validate_coarse_affine(aligner.affine_matrix, int(n_keypoints))
        aligner.compute_shifts()
        aligner.constrain_shifts()

        reg_img = palom.align.block_affine_transformed_moving_img(
            ref_img=ref_reader.read_level_channels(LEVEL, ref_channel_index),
            moving_img=mov_reader.pyramid[LEVEL],
            mxs=aligner.block_affine_matrices_da
        )
        
        log_time(f"Registered image type: {type(reg_img)}")
        log_time(f"Registered image shape: {reg_img.shape}")

        # Write OME-TIFF with proper H&E channel names and colors.
        # Process each channel individually from the dask array to avoid
        # materializing the entire (C,H,W) array and transposing it.
        he_channel_names = ["HE_Red", "HE_Green", "HE_Blue"]
        he_channel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        he_channel_metadata = [{'dye': '-', 'exposure': '-'} for _ in he_channel_names]

        # Determine number of channels and spatial shape from the dask array.
        # palom returns (C, H, W) for multi-channel or (H, W) for single-channel.
        is_dask = hasattr(reg_img, 'compute')
        reg_shape = tuple(int(d) for d in reg_img.shape)

        if len(reg_shape) == 3:
            n_ch, h, w = reg_shape
        elif len(reg_shape) == 2:
            n_ch = 1
            h, w = reg_shape
        else:
            raise ValueError(f"Unexpected registered image shape: {reg_shape}")

        # Compute each channel one at a time to keep memory usage low.
        base_planes = []
        for ch_idx in range(n_ch):
            if n_ch == 1:
                ch_data = reg_img.compute() if is_dask else np.asarray(reg_img)
            else:
                ch_slice = reg_img[ch_idx]
                ch_data = ch_slice.compute() if is_dask else np.asarray(ch_slice)
            base_planes.append(_coerce_uint16_plane(ch_data))
            log_time(f"Computed H&E channel {ch_idx + 1}/{n_ch}")

        # If only 1 channel (grayscale), replicate to RGB
        if n_ch == 1:
            base_planes = [base_planes[0]] * 3

        def registered_plane_factory(level, level_shape):
            target_height, target_width = level_shape
            for ch_idx, ch_data in enumerate(base_planes):
                if level == 0 and ch_data.shape == (target_height, target_width):
                    plane = ch_data
                else:
                    plane = _coerce_uint16_plane(
                        cv2.resize(ch_data, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    )
                if level == 0:
                    log_time(f"Wrote H&E channel: {he_channel_names[ch_idx]}")
                yield plane

        _write_ome_pyramid_from_plane_factory(
            output_path=registered_he_output,
            plane_factory=registered_plane_factory,
            channel_names=he_channel_names,
            channel_colors=he_channel_colors,
            pixel_size=ref_reader.pixel_size,
            base_shape=(h, w),
            channel_metadata=he_channel_metadata,
            n_pyramid_levels=3,
        )
        
        log_time(f"H&E registered with proper OME metadata: {registered_he_output}")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_time(f"Registration error: {e}\n{error_details}")
        # Re-raise: swallowing this leaves no registered file, so the caller marches on
        # and the failure resurfaces later as an unrelated "Merge input does not exist".
        raise


def save_merged_pyramidal_ome_tiff(codex_path, he_ome_tiff_path, output_path, mpp, channel_colors_override=None, channels_to_keep=None, overwrite=False):
    """
    Merge CODEX and H&E images with memory-efficient processing.
    Supports QPTIFF and OME-TIFF inputs.
    """
    if check_file_exists(output_path, "Merged OME-TIFF generation", overwrite=overwrite):
        return

    log_time("Starting merge of CODEX and H&E images...")
    c_names, c_colors, _c_meta, _ = get_channel_info(codex_path)
    color_map = {}
    if channel_colors_override and len(channel_colors_override) == len(c_names):
        color_map = {name: color for name, color in zip(c_names, channel_colors_override)}
    elif channel_colors_override:
        log_time(
            f"Warning: Color override length ({len(channel_colors_override)}) does not match channel count ({len(c_names)}). Using defaults."
        )

    codex_config = {
        'keep': channels_to_keep,
        'rename': {},
        'colors': color_map,
    }
    merge_multi_round_ome_tiff(
        [codex_path, he_ome_tiff_path],
        output_path,
        mpp['codex'],
        channel_configs=[codex_config, None],
        overwrite=overwrite,
    )


def create_mask_from_geometry(geometry, width, height, offset_x, offset_y):
    """Create a binary mask from GeoJSON geometry."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        if geometry['type'] == 'Polygon':
            rings = geometry['coordinates']
            if not rings: return mask
            # Exterior
            ext = np.array(rings[0]) - [offset_x, offset_y]
            cv2.fillPoly(mask, [ext.astype(np.int32)], 1)
            # Holes
            for ring in rings[1:]:
                hole = np.array(ring) - [offset_x, offset_y]
                cv2.fillPoly(mask, [hole.astype(np.int32)], 0)
                
        elif geometry['type'] == 'MultiPolygon':
            for poly_coords in geometry['coordinates']:
                if not poly_coords: continue
                rings = poly_coords
                # Exterior
                ext = np.array(rings[0]) - [offset_x, offset_y]
                cv2.fillPoly(mask, [ext.astype(np.int32)], 1)
                # Holes
                for ring in rings[1:]:
                    hole = np.array(ring) - [offset_x, offset_y]
                    cv2.fillPoly(mask, [hole.astype(np.int32)], 0)
    except Exception as e:
        log_time(f"Error creating mask from geometry: {e}")
        # Fallback to full mask if geometry fails
        mask.fill(1)
                
    return mask


def process_single_image_rois(image_path, geojson_path, rois_dir, mpp, channel_names=None, channel_colors=None, channel_metadata=None, progress_callback=None, channel_config=None):
    """
    Process ROIs for a single image (OME-TIFF or QPTIFF).
    """
    log_time(f"Processing ROIs for: {image_path}")
    
    # 1. Get Channel Info if not provided
    orig_channel_names = None
    if channel_names is None or channel_colors is None:
        try:
            c_names, c_colors, c_meta, detected_mpp = get_channel_info(image_path)
            orig_channel_names = c_names # Store original names for mapping later
            if channel_names is None:
                channel_names = c_names
            if channel_colors is None:
                channel_colors = c_colors
            if channel_metadata is None:
                channel_metadata = c_meta
        except Exception as e:
            log_time(f"Error getting channel info: {e}")
            return
    else:
        # If channel_names was provided, we still need the raw file channel names for mapping
        try:
            orig_channel_names, _, _, _ = get_channel_info(image_path)
        except Exception as e:
            log_time(f"Error getting original channel info: {e}")
            return

    # Apply channel configuration (filtering, renaming, recoloring)
    if channel_config:
        keep_list = channel_config.get('keep')
        rename_map = channel_config.get('rename', {})
        color_map = channel_config.get('colors', {})
        
        new_names = []
        new_colors = []
        new_meta = []
        
        # Filter and transform
        for i, name in enumerate(channel_names):
            if keep_list is None or name in keep_list:
                # Determine new name
                final_name = rename_map.get(name, name)
                new_names.append(final_name)
                
                # Determine new color
                if final_name in color_map:
                    new_colors.append(color_map[final_name])
                elif name in color_map:
                    new_colors.append(color_map[name])
                else:
                    new_colors.append(channel_colors[i])
                    
                new_meta.append(channel_metadata[i])
        
        channel_names = new_names
        channel_colors = new_colors
        channel_metadata = new_meta
        
        log_time(f"Applied channel config. Keeping {len(channel_names)} channels.")

    # 2. Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    if not geojson_data:
        log_time(f"Could not load GeoJSON: {geojson_path}")
        return
        
    # Use geometry extraction instead of just labels
    rois = extract_rois_with_geometry_from_geojson(geojson_data)
    if not rois:
        log_time("No ROIs found in GeoJSON.")
        return

    total_rois = len(rois)
    if progress_callback:
        progress_callback(0, total_rois)

    # 3. Prepare for reading
    try:
        with tifffile.TiffFile(image_path) as tif:
            if tif.series:
                primary_series = max(tif.series, key=lambda series: int(np.prod(getattr(series, 'shape', (0,)))))
                axes = getattr(primary_series, 'axes', None)
                if axes and 'Y' in axes and 'X' in axes:
                    img_height = int(primary_series.shape[axes.index('Y')])
                    img_width = int(primary_series.shape[axes.index('X')])
                else:
                    base_page = tif.pages[0]
                    img_height, img_width = base_page.shape[:2]
            else:
                base_page = tif.pages[0]
                img_height, img_width = base_page.shape[:2]
            
            # 4. Process ROIs
            for roi_idx, ((x1, y1, x2, y2), label, geometry) in enumerate(rois):
                # Sanitize label for filename
                safe_label = "".join([c if c.isalnum() or c in ('-','_') else '_' for c in label])
                
                # Construct output filename
                stem = Path(image_path).stem
                if stem.endswith('.ome'):
                    stem = stem[:-4]
                if stem.endswith('_merged'):
                    stem = stem[:-7]
                    
                roi_output_path = os.path.join(rois_dir, f"{stem}_{safe_label}.ome.tiff")
                
                if check_file_exists(roi_output_path, f"ROI {label}"):
                    if progress_callback:
                        progress_callback(roi_idx + 1, total_rois)
                    continue
                
                roi_width = x2 - x1
                roi_height = y2 - y1
                
                if roi_width <= 0 or roi_height <= 0:
                    log_time(f"Invalid ROI dimensions for {label}. Skipping.")
                    if progress_callback:
                        progress_callback(roi_idx + 1, total_rois)
                    continue
                
                # Ensure ROI coordinates are within image bounds
                x1 = max(0, min(x1, img_width-1))
                y1 = max(0, min(y1, img_height-1))
                x2 = max(x1+1, min(x2, img_width))
                y2 = max(y1+1, min(y2, img_height))
                roi_width = x2 - x1
                roi_height = y2 - y1
                
                # Create mask from geometry
                mask = create_mask_from_geometry(geometry, roi_width, roi_height, x1, y1)
                
                # Create OME-XML
                ome_xml = create_ome_metadata(
                    channel_names=channel_names,
                    channel_colors=channel_colors,
                    SizeX=roi_width,
                    SizeY=roi_height,
                    pixel_type='uint16',
                    n_channels=len(channel_names),
                    physical_size_x=mpp['codex'],
                    physical_size_y=mpp['codex'],
                    channel_metadata=channel_metadata
                )
                
                tile_size = (256, 256)
                n_pyramid_levels = 3
                
                with tifffile.TiffWriter(roi_output_path, bigtiff=True) as tiff_writer:
                    for i, ch_name in enumerate(channel_names):
                        # Determine page index
                        # Note: We need to map back to original names if renamed
                        # But channel_page_map uses original names from file
                        # So we need to find the original name corresponding to ch_name
                        # If we renamed, ch_name is the new name.
                        # This is tricky. We should probably keep track of original names.
                        
                        # Re-logic for channel mapping:
                        # We iterate over channel_names which are the FINAL names.
                        # We need to find the source page.
                        # If we filtered/renamed, we need to know which original index/name this corresponds to.
                        
                        # Let's assume channel_names order matches the filtered list order.
                        # We need to reconstruct the mapping or just use the index if we trust the order.
                        # Better: In the filtering step, store the original index/name.
                        
                        # Quick fix: We'll rely on the fact that we iterated over original channel_names
                        # and built the new lists. So we can just find the original name/index.
                        # But wait, we don't have the original list here anymore.
                        
                        # Let's re-get the original info to map correctly
                        # orig_c_names, _, _, _ = get_channel_info(image_path) # CAUSES LOG SPAM
                        orig_c_names = orig_channel_names
                        
                        # Find which original channel this corresponds to
                        # If we have a rename map, we can try to reverse it, but it might be ambiguous.
                        # Instead, let's look at how we built channel_names.
                        # We iterated orig_c_names.
                        # So we can just iterate orig_c_names again and apply the same logic to find the match.
                        
                        target_channel_idx = -1
                        current_new_idx = 0
                        found = False
                        
                        for orig_idx, orig_name in enumerate(orig_c_names):
                            # Apply same filter logic
                            if channel_config and channel_config.get('keep') and orig_name not in channel_config.get('keep'):
                                continue
                            
                            # This is the (current_new_idx)-th channel in the new list
                            if current_new_idx == i:
                                target_channel_idx = orig_idx
                                found = True
                                break
                            current_new_idx += 1
                            
                        if not found:
                            target_channel_idx = i
                        
                        try:
                            # Reuse the handle opened above: re-opening the stack per
                            # channel costs more than decoding the crop itself.
                            roi_data = read_channel_from_open_tif(tif, target_channel_idx, region=(y1, y2, x1, x2))
                            if roi_data.shape == mask.shape:
                                roi_data = roi_data * mask
                            else:
                                log_time(f"Warning: Mask shape {mask.shape} != ROI shape {roi_data.shape}")
                            
                            # Write base resolution
                            tiff_writer.write(
                                roi_data,
                                subifds=n_pyramid_levels,
                                photometric='minisblack',
                                compression='deflate',
                                tile=tile_size,
                                description=ome_xml if i == 0 else None,
                                planarconfig='CONTIG'
                            )
                            
                            # Generate and write pyramid levels
                            prev_level = roi_data
                            for lvl in range(n_pyramid_levels):
                                scale = 2 ** (lvl + 1)
                                new_width = max(1, roi_width // scale)
                                new_height = max(1, roi_height // scale)
                                downsampled = cv2.resize(prev_level, (new_width, new_height), interpolation=cv2.INTER_AREA)
                                tiff_writer.write(
                                    downsampled,
                                    photometric='minisblack',
                                    compression='deflate',
                                    tile=tile_size,
                                    planarconfig='CONTIG'
                                )
                                prev_level = downsampled
                                
                        except Exception as e:
                            log_time(f"Error processing channel {ch_name} for ROI {label}: {e}")
                            
                log_time(f"Completed ROI {label} -> {roi_output_path}")
                if progress_callback:
                    progress_callback(roi_idx + 1, total_rois)

    except Exception as e:
        log_time(f"Error processing image {image_path}: {e}")
        import traceback
        log_time(traceback.format_exc())


def process_rois(merged_dir, geojson_dir, rois_dir, mpp, channel_config=None):
    """
    Batch process ROIs for all merged images (OME-TIFF) or raw CODEX images (QPTIFF) in a directory.
    
    Args:
        merged_dir (str): Directory containing images
        geojson_dir (str): Directory containing GeoJSON boundary files
        rois_dir (str): Output directory for ROI images
        mpp (dict): Dictionary containing pixel sizes
        channel_config (dict): Optional configuration for channel filtering/renaming
    """
    log_time("Starting ROI extraction process...")
    
    # Find both merged OME-TIFFs and raw QPTIFFs
    all_files = os.listdir(merged_dir)
    target_files = [
        f for f in all_files
        if f.lower().endswith('_merged.ome.tiff')
        or f.lower().endswith('_merged.ome.tif')
        or f.lower().endswith('.qptiff')
        or (f.lower().endswith(('.ome.tiff', '.ome.tif')) and is_supported_image_file(f))
    ]
    
    log_time(f"Found {len(target_files)} images to process.")
    
    for filename in target_files:
        # Determine base name for matching GeoJSON
        stem = normalize_image_stem(filename)
        base_name = stem[:-7] if stem.endswith('_merged') else stem
            
        # Accept either the pipeline's own naming or a plain "<image>.geojson" exported
        # straight from QuPath, which is what the ROI tab already matches against.
        candidates = [
            f"{base_name}_roi_bounding_boxes_coords.geojson",
            f"{base_name}.geojson",
        ]
        geojson_path = next(
            (os.path.join(geojson_dir, name) for name in candidates
             if os.path.exists(os.path.join(geojson_dir, name))),
            None,
        )
        image_path = os.path.join(merged_dir, filename)

        if geojson_path is None:
            log_time(f"No geojson file found for {filename} (looked for {' or '.join(candidates)}), skipping.")
            continue
            
        process_single_image_rois(image_path, geojson_path, rois_dir, mpp, channel_config=channel_config)

    log_time("ROI extraction process completed.")


def register_codex_to_codex(
    fixed_dapi_path,
    moving_dapi_path,
    moving_codex_path,
    output_path,
    mpp,
    overwrite=False,
    fixed_reference_channel=None,
    moving_reference_channel=None,
    fixed_reference_channel_index=None,
    moving_reference_channel_index=None,
    n_keypoints=10000,
    pre_rotation_deg_moving=0.0,
):
    """
    Register a CODEX image (moving) to another CODEX image (fixed) using selected reference channels.
    Applies the transformation to all channels of the moving CODEX image.
    
    Args:
        fixed_dapi_path (str): Path to fixed reference image
        moving_dapi_path (str): Path to moving reference image
        moving_codex_path (str): Path to moving CODEX image (multi-channel)
        output_path (str): Output path for registered CODEX image (OME-TIFF)
        mpp (dict): Dictionary containing pixel sizes
        overwrite (bool): Force overwrite if file exists
    """
    if check_file_exists(output_path, "CODEX-CODEX registration", overwrite=overwrite):
        return

    log_time(f"Starting CODEX-CODEX registration: {Path(moving_codex_path).name} -> {Path(fixed_dapi_path).name}")
    
    try:
        # 1. Calculate Registration Transform using DAPI
        THUMBNAIL_LEVEL = 2
        LEVEL = 0
        
        fixed_ref_index, fixed_ref_name = resolve_reference_channel_from_image(
            fixed_dapi_path,
            reference_channel=fixed_reference_channel,
            reference_channel_index=fixed_reference_channel_index,
        )
        moving_ref_index, moving_ref_name = resolve_reference_channel_from_image(
            moving_dapi_path,
            reference_channel=moving_reference_channel,
            reference_channel_index=moving_reference_channel_index,
        )
        log_time(
            f"Using fixed reference channel: {fixed_ref_name or fixed_ref_index}; "
            f"moving reference channel: {moving_ref_name or moving_ref_index}"
        )

        # Readers for reference channels
        ref_reader = palom.reader.OmePyramidReader(fixed_dapi_path)
        ref_reader._pixel_size = mpp.get('codex_ref', mpp.get('codex', 0.3775))
        
        mov_reader = palom.reader.OmePyramidReader(moving_dapi_path)
        mov_reader._pixel_size = mpp.get('codex_mov', mpp.get('codex', 0.3775))
        
        # Thumbnails for coarse alignment
        ref_thumb = ref_reader.read_level_channels(THUMBNAIL_LEVEL, fixed_ref_index).compute()
        mov_thumb = mov_reader.read_level_channels(THUMBNAIL_LEVEL, moving_ref_index).compute()

        # Optional pre-rotation of moving reference to help coarse alignment
        # Determine if the requested angle is (approximately) a multiple of 90°.
        # Multiples of 90° are handled with np.rot90, which is exact, lossless, and has
        # no image-size limit.  Arbitrary angles use cv2.warpAffine (same-size, clips
        # corners for very large images, acceptable for keypoint matching only).
        _rot_nearest_k = round(pre_rotation_deg_moving / 90.0)
        _is_ortho_rot = abs(pre_rotation_deg_moving - _rot_nearest_k * 90.0) < 0.5
        _rot_k = int(_rot_nearest_k) % 4  # used when _is_ortho_rot is True

        if abs(pre_rotation_deg_moving) > 0.01:
            log_time(f"Applying pre-rotation of {pre_rotation_deg_moving:.1f}° to moving reference for alignment...")
            mov_base_np = np.asarray(mov_reader.read_level_channels(LEVEL, moving_ref_index).compute())
            # Ensure 2-D (squeeze single-channel dimension if present)
            if mov_base_np.ndim == 3 and mov_base_np.shape[0] == 1:
                mov_base_np = mov_base_np[0]
            mov_h, mov_w = int(mov_base_np.shape[-2]), int(mov_base_np.shape[-1])

            if _is_ortho_rot:
                # np.rot90: exact 90°-family rotation, no OpenCV size limit.
                mov_rotated_np = np.rot90(mov_base_np, k=_rot_k)
                _pre_ds_scale = 1.0
            else:
                # Arbitrary angle: cv2.warpAffine asserts BOTH src AND dst dims < SHRT_MAX.
                # Downsample to a safe size first if either dimension exceeds the limit.
                # The same scale is applied to the thumbnail so moving_thumbnail_down_factor
                # (the ratio between the two images) is preserved unchanged.
                if mov_w >= _OPENCV_MAX_DIM or mov_h >= _OPENCV_MAX_DIM:
                    _pre_ds_scale = (_OPENCV_MAX_DIM - 1) / float(max(mov_w, mov_h))
                    _ds_w = max(1, int(mov_w * _pre_ds_scale))
                    _ds_h = max(1, int(mov_h * _pre_ds_scale))
                    mov_base_np = cv2.resize(mov_base_np, (_ds_w, _ds_h),
                                             interpolation=cv2.INTER_AREA)
                else:
                    _pre_ds_scale = 1.0
                    _ds_w, _ds_h = mov_w, mov_h

                M_rot_2x3 = cv2.getRotationMatrix2D((_ds_w / 2.0, _ds_h / 2.0),
                                                     float(pre_rotation_deg_moving), 1.0)
                mov_rotated_np = cv2.warpAffine(mov_base_np, M_rot_2x3, (_ds_w, _ds_h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)

            mov_img_for_aligner = da.from_array(mov_rotated_np, chunks=(4096, 4096))

            # Rotate thumbnail with the same strategy for dimensional consistency.
            th_np = np.squeeze(np.asarray(mov_thumb))
            th_h, th_w = int(th_np.shape[-2]), int(th_np.shape[-1])
            if _is_ortho_rot:
                mov_thumb_for_aligner = np.rot90(th_np, k=_rot_k)
            else:
                # Downsample thumbnail by the same ratio as the full-res image.
                if _pre_ds_scale < 1.0:
                    _th_ds_w = max(1, int(th_w * _pre_ds_scale))
                    _th_ds_h = max(1, int(th_h * _pre_ds_scale))
                    th_np = cv2.resize(th_np, (_th_ds_w, _th_ds_h), interpolation=cv2.INTER_AREA)
                    th_h, th_w = _th_ds_h, _th_ds_w
                M_th = cv2.getRotationMatrix2D((th_w / 2.0, th_h / 2.0),
                                               float(pre_rotation_deg_moving), 1.0)
                mov_thumb_for_aligner = cv2.warpAffine(th_np, M_th, (th_w, th_h),
                                                       flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT)
        else:
            mov_img_for_aligner = mov_reader.read_level_channels(LEVEL, moving_ref_index)
            mov_thumb_for_aligner = mov_thumb
            mov_h, mov_w = None, None  # will not be needed
            _pre_ds_scale = 1.0

        aligner = palom.align.Aligner(
            ref_img=ref_reader.read_level_channels(LEVEL, fixed_ref_index),
            moving_img=mov_img_for_aligner,
            ref_thumbnail=ref_thumb,
            moving_thumbnail=mov_thumb_for_aligner,
            ref_thumbnail_down_factor=ref_reader.level_downsamples[THUMBNAIL_LEVEL] / ref_reader.level_downsamples[LEVEL],
            moving_thumbnail_down_factor=mov_reader.level_downsamples[THUMBNAIL_LEVEL] / mov_reader.level_downsamples[LEVEL]
        )

        log_time("Performing coarse affine registration...")
        aligner.coarse_register_affine(n_keypoints=int(n_keypoints))
        _validate_coarse_affine(aligner.affine_matrix, int(n_keypoints))
        aligner.compute_shifts()
        aligner.constrain_shifts()

        log_time("Registration parameters calculated.")

        # Compose block affines with inverse rotation when pre-rotation was applied.
        # The block affines map fixed → pre-rotated-moving; composing with M_rot_inv
        # gives fixed → original-moving, which is what we need for the full CODEX stack.
        if abs(pre_rotation_deg_moving) > 0.01:
            if _is_ortho_rot:
                # Use the exact pixel-coordinate inverse for np.rot90.
                M_rot_inv_3x3 = _rot90_inv_matrix(_rot_k, mov_h, mov_w)
            else:
                # For arbitrary angles the block affines from palom are in
                # DOWNSAMPLED rotated space (when _pre_ds_scale < 1).  The inverse
                # must therefore both un-rotate and upsample back to full resolution.
                # Derivation: M_inv_full = Scale(1/s) @ Unrotate_ds
                # where scale s = _pre_ds_scale.  After simplification the translation
                # terms are identical to a plain unrotation around the ORIGINAL centre;
                # only the rotation coefficients pick up the 1/s factor.
                import math as _math
                _angle_rad = _math.radians(pre_rotation_deg_moving)
                _cos_a = _math.cos(_angle_rad)
                _sin_a = _math.sin(_angle_rad)
                cx_orig, cy_orig = mov_w / 2.0, mov_h / 2.0
                M_rot_inv_3x3 = np.array([
                    [_cos_a / _pre_ds_scale, -_sin_a / _pre_ds_scale,
                     (1 - _cos_a) * cx_orig + _sin_a * cy_orig],
                    [_sin_a / _pre_ds_scale, _cos_a / _pre_ds_scale,
                     -_sin_a * cx_orig + (1 - _cos_a) * cy_orig],
                    [0, 0, 1]
                ], dtype=float)
            # block_affine_matrices_da is packed as a (3*n_r, 3*n_c) array
            # (chunks=3) where each 3×3 slice is a full affine matrix for
            # one block.  Iterate by grid position, not scalar index.
            block_mxs_np = np.asarray(aligner.block_affine_matrices_da)
            n_blk_r, n_blk_c = aligner.grid_shape
            composed_mxs = np.empty_like(block_mxs_np)
            for r in range(n_blk_r):
                for c in range(n_blk_c):
                    M_blk_3x3 = block_mxs_np[3*r:3*r+3, 3*c:3*c+3]
                    composed_mxs[3*r:3*r+3, 3*c:3*c+3] = M_rot_inv_3x3 @ M_blk_3x3
            use_block_mxs = da.from_array(composed_mxs, chunks=3)
        else:
            use_block_mxs = aligner.block_affine_matrices_da
        
        # 2. Apply Transform to All Channels of Moving CODEX
        # We need to identify channels in the moving CODEX image
        c_names, c_colors, c_meta, _ = get_channel_info(moving_codex_path)
        log_time(f"Applying transform to {len(c_names)} channels...")
        
        # Prepare output metadata
        # We use the reference image dimensions for the output
        ref_h, ref_w = ref_reader.read_level_channels(LEVEL, fixed_ref_index).shape
        
        ome_xml = create_ome_metadata(
            channel_names=c_names,
            channel_colors=c_colors,
            SizeX=ref_w,
            SizeY=ref_h,
            pixel_type='uint16',
            n_channels=len(c_names),
            physical_size_x=ref_reader.pixel_size,
            physical_size_y=ref_reader.pixel_size,
            channel_metadata=c_meta
        )
        
        # Helper to read a specific channel from moving image
        # We use a generator or iterator to handle memory
        def channel_generator(path, names):
            with tifffile.TiffFile(path) as tif:
                # Map names to pages
                name_to_page = {}
                # Try to map pages. For QPTIFF, use description. For OME-TIFF, assume order or OME-XML.
                # Simple approach: Iterate all pages, check description/tags
                # If that fails, assume sequential if count matches
                
                primary_series = None
                if tif.series:
                    primary_series = max(tif.series, key=lambda s: np.prod(getattr(s, 'shape', (0, 0))))
                
                if primary_series:
                    valid_pages = [p.index for p in primary_series.pages]
                else:
                    valid_pages = []
                    for i, p in enumerate(tif.pages):
                        if p.shape[0] > 1000: # Assume data pages are large
                            valid_pages.append(i)
                
                if len(valid_pages) == len(names):
                    for i, name in enumerate(names):
                        yield tif.pages[valid_pages[i]].asarray()
                else:
                    # Fallback: Try to find by name in description
                    for name in names:
                        found = False
                        for i in valid_pages:
                            if name in tif.pages[i].description:
                                yield tif.pages[i].asarray()
                                found = True
                                break
                        if not found:
                            log_time(f"Warning: Could not find channel {name}, returning zeros.")
                            yield np.zeros((100, 100), dtype=np.uint16) # Dummy

        # We need a way to wrap the moving channel as a dask array for palom
        # palom.align.block_affine_transformed_moving_img expects a dask array or array-like
        
        tile_size = (256, 256)
        n_pyramid_levels = 3
        
        with tifffile.TiffWriter(output_path, bigtiff=True) as tiff_writer:
            # We need to read channels one by one, wrap in dask, transform, write
            
            # Re-open file to read pages
            with tifffile.TiffFile(moving_codex_path) as tif:
                # Identify pages
                primary_series = None
                if tif.series:
                    primary_series = max(tif.series, key=lambda s: np.prod(getattr(s, 'shape', (0, 0))))
                
                if primary_series:
                    valid_pages = [p.index for p in primary_series.pages]
                else:
                    valid_pages = []
                    for i, p in enumerate(tif.pages):
                        # Filter thumbnails/labels based on size or metadata
                        # QPTIFFs have thumbnails.
                        if p.shape[0] > 500 and p.shape[1] > 500: # Heuristic
                            valid_pages.append(i)
                
                # If we have OME-XML, we might trust it more, but for now let's use the heuristic + count
                if len(valid_pages) != len(c_names):
                    log_time(f"Warning: Found {len(valid_pages)} valid pages but {len(c_names)} channels. Mapping might be incorrect.")
                
                for i, channel_name in enumerate(c_names):
                    page_idx = valid_pages[i] if i < len(valid_pages) else valid_pages[0]
                    
                    # Read image as dask array
                    # We can use da.from_array(tif.pages[page_idx].asarray()) but that reads into memory first
                    # Better: use da.from_delayed or similar, but tifffile.asarray() is usually fine for one channel if < 30GB RAM
                    
                    try:
                        img_data = tif.pages[page_idx].asarray()
                        img_da = da.from_array(img_data, chunks=tile_size)
                        
                        # Transform
                        reg_ch = palom.align.block_affine_transformed_moving_img(
                            ref_img=ref_reader.read_level_channels(LEVEL, 0),
                            moving_img=img_da,
                            mxs=use_block_mxs
                        )
                        
                        # Compute and write
                        # We compute directly to numpy for writing
                        reg_ch_np = reg_ch.compute()
                        
                        # Ensure uint16
                        reg_ch_np = reg_ch_np.astype(np.uint16)
                        
                        # Write base
                        tiff_writer.write(
                            reg_ch_np,
                            subifds=n_pyramid_levels,
                            photometric='minisblack',
                            compression='deflate',
                            tile=tile_size,
                            description=ome_xml if i == 0 else None,
                            planarconfig='CONTIG'
                        )
                        
                        # Pyramids
                        prev = reg_ch_np
                        for lvl in range(n_pyramid_levels):
                            scale = 2 ** (lvl + 1)
                            nw = max(1, ref_w // scale)
                            nh = max(1, ref_h // scale)
                            down = cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA)
                            tiff_writer.write(
                                down,
                                photometric='minisblack',
                                compression='deflate',
                                tile=tile_size,
                                planarconfig='CONTIG'
                            )
                            prev = down
                            
                        log_time(f"Registered channel {i+1}/{len(c_names)}: {channel_name}")
                        
                    except Exception as e:
                        log_time(f"Error registering channel {channel_name}: {e}")
                        
        log_time(f"CODEX-CODEX registration complete: {output_path}")
        
    except Exception as e:
        log_time(f"Registration failed: {e}")
        import traceback
        log_time(traceback.format_exc())


def merge_multi_round_ome_tiff(image_paths, output_path, mpp, channel_configs=None, overwrite=False):
    """
    Merge multiple OME-TIFF/QPTIFF images into a single OME-TIFF stack.
    Handles channel selection and renaming.
    
    Args:
        image_paths (list): List of file paths to merge
        output_path (str): Output file path
        mpp (float): Microns per pixel for metadata
        channel_configs (list of dict): Configuration for each image.
            Each dict should have:
            - 'keep': list of channel names to keep (None for all)
            - 'rename': dict of {old_name: new_name}
        overwrite (bool): Force overwrite if file exists
    """
    if check_file_exists(output_path, "Multi-round merge", overwrite=overwrite):
        return

    log_time(f"Starting multi-round merge of {len(image_paths)} images...")
    
    try:
        # 1. Collect all channels to be written
        final_channels = [] # List of (image_index, channel_index, original_name, new_name, color, metadata)
        
        ref_shape = None
        
        for img_idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Merge input does not exist: {img_path}")

            c_names, c_colors, c_meta, _ = get_channel_info(img_path)
            if not c_names:
                raise ValueError(f"No channels were detected for merge input: {img_path}")
            
            # Get config for this image
            config = channel_configs[img_idx] if channel_configs and img_idx < len(channel_configs) else {}
            if config is None:
                config = {}
            
            # Index-based filtering (from GUI) — handles duplicate channel names correctly.
            keep_indices = config.get('keep_indices')
            keep_idx_set = set(keep_indices) if keep_indices is not None else None
            rename_by_index = config.get('rename_by_index', {})
            colors_by_index = config.get('colors_by_index', {})
            # Legacy name-based filtering (backward compat with notebooks)
            keep_list = config.get('keep')
            rename_map = config.get('rename', {})
            color_map = config.get('colors', {})
            
            # Check shape of first image to set reference
            if img_idx == 0:
                ref_shape = _spatial_shape_from_image(img_path)
            
            for ch_idx, name in enumerate(c_names):
                # Filter: index-based takes precedence over name-based
                if keep_idx_set is not None:
                    if ch_idx not in keep_idx_set:
                        continue
                elif keep_list is not None:
                    if name not in keep_list:
                        continue
                
                # Rename: index-based takes precedence
                if ch_idx in rename_by_index:
                    new_name = rename_by_index[ch_idx]
                else:
                    new_name = rename_map.get(name, name)
                
                # Color: index-based takes precedence
                if ch_idx in colors_by_index:
                    final_color = colors_by_index[ch_idx]
                else:
                    final_color = color_map.get(name, c_colors[ch_idx])
                
                final_channels.append({
                    'img_idx': img_idx,
                    'ch_idx': ch_idx,
                    'name': new_name,
                    'color': final_color,
                    'meta': c_meta[ch_idx]
                })

        if not final_channels:
            raise ValueError("No channels remained after applying merge configuration.")

        log_time(f"Total channels to merge: {len(final_channels)}")

        # Pyramid-level cache: pre-compute levels 1-3 while writing level 0 to avoid
        # re-reading large source files 3 extra times per channel. Held compressed
        # because all channels are cached before the level-1 pass begins; zstd-1 keeps
        # that round trip roughly ten times cheaper than zlib at a better ratio, and
        # decompresses to the same bytes.
        _pyr_cache = {}  # {(ch_idx, level): (shape, dtype, codec, payload, nbytes)}

        def merged_plane_factory(level, level_shape):
            target_height, target_width = level_shape
            for i, ch_info in enumerate(final_channels):
                # Serve from cache for pyramid levels
                if level > 0:
                    cache_key = (i, level)
                    if cache_key in _pyr_cache:
                        shape, dtype, codec, payload, nbytes = _pyr_cache.pop(cache_key)
                        yield _unpack_cached_plane(codec, payload, nbytes, shape, dtype)
                        continue

                # Read base channel from source file
                img_path = image_paths[ch_info['img_idx']]
                ch_idx = ch_info['ch_idx']
                img = read_channel_image(img_path, ch_idx)

                if img.shape != ref_shape:
                    img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)

                base_plane = _coerce_uint16_plane(img)

                if level == 0:
                    # Pre-compute all pyramid levels and store compressed in cache
                    for cache_level in range(1, 4):
                        scale = 2 ** cache_level
                        cl_h = max(1, ref_shape[0] // scale)
                        cl_w = max(1, ref_shape[1] // scale)
                        ds = _coerce_uint16_plane(
                            cv2.resize(base_plane, (cl_w, cl_h), interpolation=cv2.INTER_AREA)
                        )
                        try:
                            codec, payload, nbytes = _pack_cached_plane(ds)
                            _pyr_cache[(i, cache_level)] = (
                                ds.shape, ds.dtype, codec, payload, nbytes
                            )
                        except MemoryError:
                            _pyr_cache.clear()  # Free memory; fall back to disk re-reads
                    log_time(f"Merged channel {i+1}/{len(final_channels)}: {ch_info['name']}")
                    yield base_plane
                else:
                    # Cache miss — re-read and resize as fallback
                    resized = cv2.resize(base_plane, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    yield _coerce_uint16_plane(resized)

        _write_ome_pyramid_from_plane_factory(
            output_path=output_path,
            plane_factory=merged_plane_factory,
            channel_names=[c['name'] for c in final_channels],
            channel_colors=[c['color'] for c in final_channels],
            pixel_size=mpp,
            base_shape=ref_shape,
            channel_metadata=[c['meta'] for c in final_channels],
            n_pyramid_levels=3,
        )

        log_time(f"Multi-round merge complete: {output_path}")

    except Exception as e:
        log_time(f"Merge failed: {e}")
        import traceback
        log_time(traceback.format_exc())


