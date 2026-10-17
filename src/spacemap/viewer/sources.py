"""Uniform, tile-level access to the pyramidal images SPACEMAP produces and reads.

Two on-disk layouts cover every image in a project:

* **Series pyramids** -- QPTIFF from the scanner (CODEX ``CYX``, H&E ``YXS``)
  and most third-party pyramidal TIFFs. ``tifffile`` exposes them as one series
  whose ``levels`` are the pyramid.
* **Channel pages with sub-resolutions** -- the OME-TIFFs SPACEMAP writes
  (registered images, ROIs): one top-level page per channel, each carrying its
  lower resolutions as SubIFDs. ``tifffile`` does not assemble these into one
  series, so they are read page by page, which is also much faster to open.

An :class:`ImageSource` hides the difference: it reports the pyramid, the
channel names and the pixel size, and reads any region of any level for any
channel without loading the rest of the image.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = (".qptiff", ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".svs")


@dataclass
class Level:
    width: int
    height: int
    downsample_x: float
    downsample_y: float


@dataclass
class ImageInfo:
    path: str
    width: int
    height: int
    levels: list[Level]
    channel_names: list[str]
    dtype: str
    rgb: bool
    pixel_size_um: float | None
    layout: str
    fingerprint: str
    extra: dict = field(default_factory=dict)
    channel_colors: list = field(default_factory=list)  # "#rrggbb" per channel as the file records it, or None
    channel_meta: list = field(default_factory=list)  # {"dye", "exposure_ms"} per channel as the file records them

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "name": Path(self.path).name,
            "width": self.width,
            "height": self.height,
            "levels": [vars(level) for level in self.levels],
            "channels": self.channel_names,
            "dtype": self.dtype,
            "rgb": self.rgb,
            "pixel_size_um": self.pixel_size_um,
            "layout": self.layout,
            "fingerprint": self.fingerprint,
            "colors": self.channel_colors,
            "meta": self.channel_meta,
        }


def fingerprint(path: str | Path) -> str:
    stat = Path(path).stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


def _channel_names_from_ome(ome_xml: str, count: int) -> list[str]:
    names = re.findall(r"<Channel\b[^>]*\bName=\"([^\"]*)\"", ome_xml or "")
    return names[:count] if len(names) >= count else []


def _channel_colors(tif, count: int) -> list:
    """Each channel's colour as the file records it ("#rrggbb", or None): OME's Channel/@Color (red, green,
    blue in its top three bytes; any alpha), else a QPTIFF page's <Color>r,g,b</Color>."""
    colours = []
    ome = tif.ome_metadata or ""
    if ome:
        for match in re.finditer(r"<Channel\b[^>]*>", ome):
            value = re.search(r'\bColor="(-?\d+)"', match.group(0))
            packed = int(value.group(1)) & 0xFFFFFFFF if value else None
            colours.append("#%02x%02x%02x" % ((packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255) if packed is not None else None)
    elif tif.is_qpi:
        for page in list(tif.pages)[:count]:
            value = re.search(r"<Color>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</Color>", page.description or "")
            colours.append("#%02x%02x%02x" % tuple(min(255, int(v)) for v in value.groups()) if value else None)
    colours = [None if colour == "#000000" else colour for colour in colours[:count]]  # black would hide the marker
    return colours + [None] * (count - len(colours))


def _exposure_ms(value, unit="s"):
    factor = {"s": 1000.0, "ms": 1.0, "µs": 0.001, "us": 0.001, "ns": 1e-6}.get(unit or "s")
    try:
        return round(float(value) * factor, 3) if factor else None
    except (TypeError, ValueError):
        return None


def _channel_meta(tif, count: int) -> list:
    """Each channel's dye and exposure (ms) as the file records them: OME's Channel/@Fluor and Plane/@ExposureTime
    (or SPACEMAP's older Channel/@Exposure in ms), else a QPTIFF page's filter <Name> and <ExposureTime> (µs)."""
    meta = []
    ome = tif.ome_metadata or ""
    if ome:
        planes = {}
        for match in re.finditer(r"<Plane\b[^>]*>", ome):
            the_c = re.search(r'\bTheC="(\d+)"', match.group(0))
            exposure = re.search(r'\bExposureTime="([^"]+)"', match.group(0))
            unit = re.search(r'\bExposureTimeUnit="([^"]+)"', match.group(0))
            if the_c and exposure:
                planes.setdefault(int(the_c.group(1)), _exposure_ms(exposure.group(1), unit.group(1) if unit else "s"))
        for index, match in enumerate(re.finditer(r"<Channel\b[^>]*>", ome)):
            tag = match.group(0)
            dye = re.search(r'\bFluor="([^"]*)"', tag)
            old = re.search(r'\bExposure="([^"]*)"', tag)
            meta.append({"dye": dye.group(1) if dye else None,
                         "exposure_ms": planes.get(index, _exposure_ms(old.group(1), "ms") if old else None)})
    elif tif.is_qpi:
        for page in list(tif.pages)[:count]:
            text = page.description or ""
            name = re.search(r"<Name>([^<]*)</Name>", text)
            exposure = re.search(r"<ExposureTime>([^<]*)</ExposureTime>", text)
            meta.append({"dye": name.group(1).strip() if name else None, "exposure_ms": _exposure_ms(exposure.group(1), "µs") if exposure else None})
    meta = meta[:count]
    return meta + [{"dye": None, "exposure_ms": None}] * (count - len(meta))


def _channel_names_from_qptiff(pages, count: int) -> list[str]:
    names = []
    for page in list(pages)[:count]:
        description = page.description or ""
        match = re.search(r"<Biomarker>([^<]*)</Biomarker>", description) or re.search(
            r"<Name>([^<]*)</Name>", description
        )
        names.append(match.group(1).strip() if match else "")
    return names if len(names) == count and any(names) else []


def _pixel_size_um(tif, page) -> float | None:
    """From OME PhysicalSizeX, else the TIFF resolution tags."""
    ome = tif.ome_metadata or ""
    match = re.search(r"PhysicalSizeX=\"([0-9.eE+-]+)\"", ome)
    if match:
        try:
            unit = re.search(r"PhysicalSizeXUnit=\"([^\"]*)\"", ome)
            value = float(match.group(1))
            if unit and unit.group(1) in ("nm",):
                value /= 1000.0
            elif unit and unit.group(1) in ("mm",):
                value *= 1000.0
            return value if value > 0 else None
        except ValueError:
            pass
    try:
        x_resolution = page.tags["XResolution"].value
        unit = int(page.tags["ResolutionUnit"].value)
        pixels_per_unit = x_resolution[0] / x_resolution[1]
        if pixels_per_unit <= 0:
            return None
        if unit == 3:  # centimetre
            return 1e4 / pixels_per_unit
        if unit == 2:  # inch
            return 25400.0 / pixels_per_unit
    except Exception:
        return None
    return None


class ImageSource:
    """One open image. Not shared between threads; see :func:`open_source`."""

    def __init__(self, path: str | Path) -> None:
        import tifffile

        self.path = str(Path(path).expanduser())
        self._stamp = fingerprint(self.path)
        self._tif = tifffile.TiffFile(self.path)
        self._arrays: dict[tuple, object] = {}
        pages = self._tif.pages
        first = pages[0]
        top_count = len(pages)
        channel_pages = (
            top_count > 1
            and bool(first.subifds)
            and first.ndim == 2
            and all(pages[i].shape == first.shape for i in (0, top_count - 1))
        )
        if channel_pages:
            self._init_channel_pages(first, top_count)
        else:
            self._init_series()

    # -- layouts ---------------------------------------------------------

    def _init_channel_pages(self, first, count: int) -> None:
        self.layout = "channel-pages"
        # Every channel has the same levels, so only the first is read now; the
        # others are read when a tile of that channel is first needed. Walking
        # all of them up front meant a thousand small reads per image on a
        # network file system (about 1.5 s) before anything could be shown.
        self._channel_level_pages: list[list | None] = [None] * count
        self._channel_level_pages[0] = self._levels_of(0)
        heights_widths = [(level.shape[0], level.shape[1]) for level in self._channel_level_pages[0]]
        self._levels_shape = heights_widths
        self.rgb = False
        self.dtype = str(first.dtype)
        names = _channel_names_from_ome(self._tif.ome_metadata or "", count)
        self.channel_names = names or [f"Channel {i + 1}" for i in range(count)]
        self._pixel = _pixel_size_um(self._tif, first)

    def _levels_of(self, channel: int) -> list:
        import tifffile

        page = self._tif.pages[channel]
        levels = [page]
        if page.subifds:
            levels.extend(tifffile.TiffPages(page))
        return levels

    def _init_series(self) -> None:
        self.layout = "series"
        series = self._tif.series[0]
        self._series = series
        axes = series.axes
        self._axes = axes
        self.rgb = axes.endswith("S") and series.shape[-1] in (3, 4)
        shapes = [level.shape for level in series.levels]
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        self._levels_shape = [(shape[y_axis], shape[x_axis]) for shape in shapes]
        channel_axis = next((axes.index(a) for a in ("C", "I", "S") if a in axes and not (a == "S" and self.rgb)), None)
        self._channel_axis = channel_axis
        count = 1 if channel_axis is None else series.shape[channel_axis]
        self.dtype = str(series.dtype)
        names = []
        if self._tif.is_qpi:
            names = _channel_names_from_qptiff(self._tif.pages, count)
        if not names:
            names = _channel_names_from_ome(self._tif.ome_metadata or "", count)
        if self.rgb:
            self.channel_names = ["RGB"]
        else:
            self.channel_names = names or [f"Channel {i + 1}" for i in range(count)]
        self._pixel = _pixel_size_um(self._tif, self._tif.pages[0])

    # -- description -------------------------------------------------------

    @property
    def info(self) -> ImageInfo:
        base_h, base_w = self._levels_shape[0]
        levels = [
            Level(width=w, height=h, downsample_x=base_w / w, downsample_y=base_h / h) for h, w in self._levels_shape
        ]
        return ImageInfo(
            path=self.path,
            width=base_w,
            height=base_h,
            levels=levels,
            channel_names=list(self.channel_names),
            channel_colors=[] if self.rgb else _channel_colors(self._tif, len(self.channel_names)),
            channel_meta=[] if self.rgb else _channel_meta(self._tif, len(self.channel_names)),
            dtype=self.dtype,
            rgb=self.rgb,
            pixel_size_um=self._pixel,
            layout=self.layout,
            fingerprint=fingerprint(self.path),
        )

    # -- reading ------------------------------------------------------------

    def _channel_page(self, level: int, channel: int):
        if level == 0:
            # Full-resolution cell measurements do not need sub-resolutions.
            return self._tif.pages[channel]
        if self._channel_level_pages[channel] is None:
            self._channel_level_pages[channel] = self._levels_of(channel)
        return self._channel_level_pages[channel][level]

    def _array(self, level: int, channel: int):
        import zarr

        key = (level, channel if self.layout == "channel-pages" else None)
        array = self._arrays.get(key)
        if array is None:
            if self.layout == "channel-pages":
                page = self._channel_page(level, channel)
                array = zarr.open(page.aszarr(), mode="r")
            else:
                # A pyramid series opens as a group with one array per level.
                group = self._arrays.get("group")
                if group is None:
                    group = self._arrays["group"] = zarr.open(self._series.aszarr(), mode="r")
                array = group if isinstance(group, zarr.Array) else group[str(level)]
            self._arrays[key] = array
        return array

    def _whole_level(self, level: int, channel: int) -> np.ndarray | None:
        """A small level of one channel, read in one call and kept in memory (None for large levels).

        Read through zarr, a level costs one file-system round trip per stored
        tile: the smallest level of PDAC's s8r1 (6,780 x 4,405 px, about 500
        tiles) took 10.6 s that way, and 0.2 s read whole. Automatic contrast
        and zoomed-out tiles read whole small levels, so this is what made
        large ROIs open slowly.
        """
        single = self.layout == "series" and self._channel_axis is None and not self.rgb  # e.g. cell-label pyramids
        if self.layout != "channel-pages" and not single:
            return None
        height, width = self._levels_shape[level]
        if height * width * np.dtype(self.dtype).itemsize > _WHOLE_LEVEL_BYTES:
            return None
        key = (self.path, self._stamp, level, channel)
        with _WHOLE_LOCK:
            hit = _WHOLE.get(key)
            if hit is not None:
                _WHOLE.move_to_end(key)
                return hit
            lock = _WHOLE_READING.setdefault(key, threading.Lock())
        with lock:  # one thread reads it; the others wait for it
            with _WHOLE_LOCK:
                hit = _WHOLE.get(key)
            if hit is not None:
                return hit
            if single:
                data = self._series.levels[level].asarray(maxworkers=8)
            else:
                data = self._channel_page(level, channel).asarray(maxworkers=8)
            data.setflags(write=False)
            with _WHOLE_LOCK:
                _WHOLE[key] = data
                _WHOLE_SIZE[0] += data.nbytes
                while _WHOLE_SIZE[0] > _WHOLE_CACHE_BYTES and len(_WHOLE) > 1:
                    _, old = _WHOLE.popitem(last=False)
                    _WHOLE_SIZE[0] -= old.nbytes
                _WHOLE_READING.pop(key, None)
            return data

    def read_region(self, level: int, channel: int, x0: int, y0: int, x1: int, y1: int, *, cache_whole=True) -> np.ndarray:
        """Pixels of one channel (or RGB) in [y0:y1, x0:x1] at *level*."""
        height, width = self._levels_shape[level]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(x1)), min(height, int(y1))
        if x1 <= x0 or y1 <= y0:
            shape = (0, 0, 3) if self.rgb else (0, 0)
            return np.zeros(shape, dtype=self.dtype)
        whole = self._whole_level(level, channel) if cache_whole else None
        if whole is not None:
            return whole[y0:y1, x0:x1]
        array = self._array(level, channel)
        if self.layout == "channel-pages":
            return np.asarray(array[y0:y1, x0:x1])
        index = [slice(None)] * len(self._axes)
        index[self._axes.index("Y")] = slice(y0, y1)
        index[self._axes.index("X")] = slice(x0, x1)
        if self._channel_axis is not None:
            index[self._channel_axis] = channel
        region = np.asarray(array[tuple(index)])
        if self.rgb and region.shape[-1] == 4:
            region = region[..., :3]
        return region

    def close(self) -> None:
        try:
            self._tif.close()
        except Exception:
            pass


# Small levels read whole (see ImageSource._whole_level), shared by all threads.
_WHOLE_LEVEL_BYTES = 96 * 1024**2  # a level up to this size is read in one call
_WHOLE_CACHE_BYTES = 1536 * 1024**2  # memory kept for such levels, across images
_WHOLE: OrderedDict = OrderedDict()
_WHOLE_SIZE = [0]
_WHOLE_LOCK = threading.Lock()
_WHOLE_READING: dict = {}

# ---------------------------------------------------------------------------
# Per-thread cache of open sources
# ---------------------------------------------------------------------------

_LOCAL = threading.local()
_MAX_OPEN_PER_THREAD = 8


def open_source(path: str | Path) -> ImageSource:
    """An open :class:`ImageSource` for this thread, reopened if the file changed.

    Each worker thread keeps its own file handles, so concurrent tile requests
    read in parallel instead of queuing on one handle's lock.
    """
    cache: OrderedDict = getattr(_LOCAL, "sources", None)
    if cache is None:
        cache = _LOCAL.sources = OrderedDict()
    key = str(Path(path).expanduser())
    stamp = fingerprint(key)
    entry = cache.get(key)
    if entry is not None and entry[0] == stamp:
        cache.move_to_end(key)
        return entry[1]
    if entry is not None:
        entry[1].close()
    source = ImageSource(key)
    cache[key] = (stamp, source)
    while len(cache) > _MAX_OPEN_PER_THREAD:
        _, (_, old) = cache.popitem(last=False)
        old.close()
    return source


_INFOS: OrderedDict = OrderedDict()
_INFO_LOCK = threading.Lock()


def image_info(path: str | Path) -> ImageInfo:
    """An image's description (size, levels, markers), read once per file version for the whole process.

    Page updates run in fresh threads, so the per-thread open files of
    open_source() would be opened again on every update; the description alone
    is all the interface needs.
    """
    key = str(Path(path).expanduser())
    stamp = fingerprint(key)
    with _INFO_LOCK:
        hit = _INFOS.get(key)
        if hit is not None and hit[0] == stamp:
            _INFOS.move_to_end(key)
            return hit[1]
    info = open_source(key).info
    with _INFO_LOCK:
        _INFOS[key] = (stamp, info)
        while len(_INFOS) > 256:
            _INFOS.popitem(last=False)
    return info


def is_viewable(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(SUPPORTED_SUFFIXES) and not name.startswith(".")
