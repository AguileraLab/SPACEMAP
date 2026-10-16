"""HTTP routes the viewer calls: image information, contrast limits and tiles.

Served on the same port as the interface (see ``spacemap_gui/server.py``), so
nothing extra has to be forwarded when SPACEMAP runs on a cluster node. Every
route takes a registry token, never a path.

    GET /image/{token}/info                     pyramid, channels, pixel size
    GET /image/{token}/limits?ch=0,5,9          suggested display limits
    GET /image/{token}/histogram?ch=3[&shape=<token>[&sid=<id>,<id>|&name=<name>]]
                                                one marker's histogram (whole image or in a shape)
    GET /image/{token}/tile/{z}/{x}/{y}.jpg     a rendered tile
        ?s=<channel>:<rrggbb>:<low>:<high>[:<gamma>];...   channel styles
        &v=<fingerprint>                                    cache key

Cell, annotation, shape and display-setting layers have routes of their own
below; each is registered by the interface with ``registry.register_object``.
"""

from __future__ import annotations

import time
from pathlib import Path

# Initialise GEOS before concurrent request workers import Shapely submodules.
import shapely

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from spacemap.viewer import render
from spacemap.viewer.sources import image_info, open_source

from . import artifact_api, classification_api, annotation_export_api, annotation_difference_api, registry

# Studio's routes run on server threads, outside any page run of the classic interface. Streamlit then warns
# "missing ScriptRunContext" or "No runtime found" whenever shared code touches one of its helpers: expected here, and
# only noise in the terminal.
import logging as _logging

for _name in ("streamlit.runtime.scriptrunner_utils.script_run_context", "streamlit.runtime.caching.cache_data_api"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

#: Set when the routes are mounted beside the interface on its own port.
MOUNTED = False
_TILE_CACHE_SECONDS = 7 * 24 * 3600


def _source_or_404(token: str):
    path = registry.resolve(token)
    if path is None:
        return None, JSONResponse({"error": "Unknown image. Reopen it from SPACEMAP."}, status_code=404)
    try:
        return (path, open_source(path)), None
    except FileNotFoundError:
        return None, JSONResponse({"error": "The image file no longer exists."}, status_code=404)
    except Exception as exc:
        return None, JSONResponse({"error": f"Could not open the image: {exc}"}, status_code=500)


def parse_styles(text: str | None) -> list[render.ChannelStyle]:
    styles = []
    for part in (text or "").split(";"):
        if not part.strip():
            continue
        fields = part.split(":")
        if len(fields) < 4:
            raise ValueError(f"Bad channel style: {part!r}")
        gamma = float(fields[4]) if len(fields) > 4 and fields[4] else 1.0
        styles.append(
            render.ChannelStyle(
                int(fields[0]), render.parse_color(fields[1]), float(fields[2]), float(fields[3]), gamma
            )
        )
    return styles


def info(request: Request) -> Response:
    from spacemap.background_detection import THRESHOLD_METHODS

    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    _, source = found
    details = source.info
    payload = details.as_dict()
    payload.pop("path", None)
    payload["tile_size"] = render.TILE
    payload["max_zoom"] = render.max_zoom(details)
    payload["auto_methods"] = list(THRESHOLD_METHODS)
    from spacemap.thresholding_engine import catalog
    payload["thresholding"] = catalog()
    return JSONResponse(payload)


def limits(request: Request) -> Response:
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    details = source.info
    try:
        channels = [int(c) for c in request.query_params.get("ch", "").split(",") if c.strip()]
    except ValueError:
        return JSONResponse({"error": "ch must be channel numbers"}, status_code=400)
    result = {}
    for channel in channels:
        if 0 <= channel < details.channel_count and not details.rgb:
            result[str(channel)] = list(render.channel_limits(path, details, channel))
    return JSONResponse(result)


def tile(request: Request) -> Response:
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    details = source.info
    try:
        z = int(request.path_params["z"])
        x = int(request.path_params["x"])
        y = int(request.path_params["y"])
        styles = parse_styles(request.query_params.get("s")) if 's' in request.query_params else None
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if z < 0 or x < 0 or y < 0 or z > render.max_zoom(details) + 1:
        return Response(status_code=404)
    started = time.perf_counter()
    lossless = request.url.path.endswith('.png')
    preview = request.query_params.get("threshold_preview")
    if preview:
        import io
        import numpy as np
        from PIL import Image
        try:
            ch, cutoff, colour, mode = preview.split(":")
            ch, cutoff = int(ch), float(cutoff)
            if details.rgb or not 0 <= ch < details.channel_count or not np.isfinite(cutoff) or mode not in {"mask", "overlay"}:
                raise ValueError("Invalid threshold preview.")
            rgb = np.asarray(render.parse_color(colour)) * 255
            values = render.raw_tile(path, details, z, x, y, ch)
            positive = np.isfinite(values) & (values > cutoff)
            if mode == "overlay":
                base = render.render_tile(path, details, z, x, y, styles, lossless=True)
                output = np.array(Image.open(io.BytesIO(base)).convert("RGB"))
                output[positive] = .45 * output[positive] + .55 * rgb
            else:
                output = np.zeros((*values.shape, 3), dtype=np.uint8)
                output[positive] = rgb
            buffer = io.BytesIO(); Image.fromarray(output).save(buffer, format="PNG")
            return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    body = render.render_tile(path, details, z, x, y, styles, lossless=lossless)
    kind = "image/png" if lossless else "image/jpeg"
    square = request.query_params.get("ck")
    if square:  # a checkerboard: every other square see-through (Registration's compare)
        if not square.isdigit() or not 8 <= int(square) <= 100000:
            return JSONResponse({"error": "Choose a square size in pixels."}, status_code=400)
        body, kind = render.checker_tile(body, details, z, x, y, int(square)), "image/png"
    headers = {
        "Cache-Control": f"private, max-age={_TILE_CACHE_SECONDS}",
        "Server-Timing": f"render;dur={(time.perf_counter() - started) * 1000:.1f}",
    }
    return Response(body, media_type=kind, headers=headers)


def _measurement_region(params, image: str):
    """Resolve registered shapes/annotation classes in this image's coordinates."""
    from shapely.geometry import shape as to_geometry

    from spacemap.viewer import annotations

    if params.get("shape") and params.get("annotation"):
        raise ValueError("Choose one measurement region.")
    if params.get("shape"):
        layer = registry.resolve_object(params["shape"], "shapes")
        if not layer or Path(layer["image"]).resolve() != Path(image).resolve():
            raise ValueError("That shape does not belong to this image.")
        from spacemap_gui.studio import annotation_edits

        features = annotation_edits.shown(layer)  # as the browser shows it, so the ids picked there are found
        # One polygon (sid), several picked ones (sid=a,b,c), all of one name (name=...), or the whole layer: measured as one.
        wanted = [item for item in params.get("sid", "").split(",") if item]
        if wanted:
            chosen = [f for f in features if str(f["properties"]["id"]) in wanted]
            if len(chosen) != len(set(wanted)):
                raise LookupError("A chosen shape no longer exists. Choose the region again.")
        elif "name" in params:
            chosen = [f for f in features if str(f["properties"].get("name") or "") == params["name"]]
            if not chosen:
                raise LookupError(f"No polygon is named {params['name']} any more. Choose the region again.")
        else:
            chosen = features
            if not chosen:
                raise LookupError("That layer has no polygons.")
        regions = [to_geometry(f["geometry"]) for f in chosen]
    elif params.get("annotation"):
        layer = registry.resolve_object(params["annotation"], "annotations")
        if not layer or Path(layer["image"]).resolve() != Path(image).resolve():
            raise ValueError("That annotation does not belong to this image.")
        loaded = annotations.load_annotations(layer["path"])
        regions = [
            g
            for g, name in zip(loaded.geometries, loaded.classes)
            if not params.get("class") or name == params["class"]
        ]
    else:
        return None
    # A self-touching outline among many pieces is repaired rather than failing the whole region.
    regions = [g if g.is_valid else g.buffer(0) for g in regions]
    regions = [g for g in regions if not g.is_empty and g.area > 0]
    if not regions:
        raise ValueError("The selected annotation has no valid area. Select another region.")
    return regions


def background(request: Request) -> Response:
    """Estimate a raw background threshold using Tab 2's detection function."""
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    try:
        channel = int(request.query_params.get("ch", ""))
        if source.info.rgb or not 0 <= channel < source.info.channel_count:
            raise ValueError("No such marker.")
        region = _measurement_region(request.query_params, path)
        result = render.background_level(
            path, source.info, channel, request.query_params.get("method", "Triangle"), region=region,
            params=dict(request.query_params), native=request.query_params.get("native") == "1"
        )
        return JSONResponse(result)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def threshold_benchmark(request: Request) -> Response:
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    q = request.query_params
    try:
        channel = int(q.get("ch", ""))
        if source.info.rgb or not 0 <= channel < source.info.channel_count:
            raise ValueError("No such marker.")
        result = render.threshold_benchmark(path, source.info, channel, q.get("methods", "Otsu,Triangle,Li,Yen").split(","),
            region=_measurement_region(q, path), params=dict(q), native=q.get("native") == "1",
            scale=q.get("scale", "Linear"), cofactor=float(q.get("cofactor", "5")))
        return JSONResponse(result)
    except (TypeError, ValueError, LookupError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def histogram(request: Request) -> Response:
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    details = source.info
    try:
        channel = int(request.query_params.get("ch", ""))
    except ValueError:
        return JSONResponse({"error": "ch must be a channel number"}, status_code=400)
    if details.rgb or not 0 <= channel < details.channel_count:
        return JSONResponse({"error": "No such marker."}, status_code=400)
    try:
        region = _measurement_region(request.query_params, path)
        q = request.query_params
        return JSONResponse(render.channel_histogram(path, details, channel, region=region,
            native=q.get("native") == "1", scale=q.get("scale", "Linear"), cofactor=float(q.get("cofactor", "5"))))
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def auto_display_range(request: Request) -> Response:
    found, error = _source_or_404(request.path_params["token"])
    if error:
        return error
    path, source = found
    q = request.query_params
    try:
        channel = int(q.get("ch", ""))
        if source.info.rgb or not 0 <= channel < source.info.channel_count:
            raise ValueError("No such marker.")
        return JSONResponse(render.auto_display_range(path, source.info, channel,
            region=_measurement_region(q, path), native=q.get("native") == "1",
            mode=q.get("mode", "both"), low=float(q.get("low", "0")),
            method=q.get("method", "Triangle"), params=dict(q),
            upper_percentile=float(q.get("upper_percentile", "99.8"))))
    except (ValueError, TypeError, LookupError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "label_batches": True, "consensus_model": True, "classification": 1, "single_tissue": 1, "annotation_layers": 1, "annotation_difference": 1, "native_image_detail": 1})


# ---------------------------------------------------------------------------
# Overlay layers: cells and annotations
# ---------------------------------------------------------------------------


def _png(rgba) -> Response:
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    if rgba.size == 0:
        rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", compress_level=1)
    return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


def _layer(request: Request, kind: str):
    layer = registry.resolve_object(request.path_params["token"], kind)
    if layer is None:
        return None, None, JSONResponse({"error": "Unknown layer. Reopen the viewer."}, status_code=404)
    if kind == "cells":
        from spacemap_gui.studio.cell_links import table_layer
        layer = table_layer(layer, request.query_params.get("t", ""))
    if kind == "annotations":
        path = Path(layer.get("path", ""))
        if not path.is_file():
            message = ("This annotation entry is a folder, not a GeoJSON file. "
                       "Reopen the image to refresh its layers; keep the History folder.") if path.is_dir() else (
                       "This annotation file is no longer available. Reopen the image to refresh its layers.")
            return None, None, JSONResponse({"error": message}, status_code=400)
    try:
        image = image_info(layer["image"])
    except Exception as exc:
        return None, None, JSONResponse({"error": f"Could not open the image: {exc}"}, status_code=500)
    return layer, image, None


def _sanitize(value) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "item"


_LUTS: dict = {}


def _cell_lut(layer: dict, table_id: str, column: str, hidden: frozenset, colormap: str, limits=(None, None)):
    """Colour table for a column, cached per table version and settings. *limits* (low, high) keep a numeric
    column's colours to the cells in that range (the others are not coloured) and spread the scale over it."""
    from spacemap.viewer import cells

    if not column:
        return None, None
    scale = False
    if column.startswith("preview:"):
        # Cell tagging's preview: columns computed in memory, not yet written to any table.
        from spacemap_gui.studio import tagging

        found = tagging.preview_values(layer.get("image"), layer.get("roi"), column[len(tagging.PREFIX):])
        if found is None:
            return None, None
        labels, values, version = found
        key_table = ("preview", version)
        scale = tagging.preview_is_scale(layer.get("image"), layer.get("roi"), column[len(tagging.PREFIX):])
    else:
        key_table = (layer.get("tables") or {}).get(table_id)
        if not key_table:
            return None, None
        labels, values = cells.roi_values(
            key_table,
            layer["roi"],
            column,
            roi_column=layer.get("roi_column", "batch_id"),
            label_column=layer.get("label_column", "label"),
        )
    from spacemap.phenotype_presets import signature
    key = (id(values), key_table, column, hidden, colormap, limits, signature())
    cached = _LUTS.get(key)
    if cached is not None:
        return cached
    if (column.startswith("preview:") and scale) or cells.is_numeric(values) or _numbers(values):
        low, high = limits
        if low is not None and high is not None and high < low:
            low, high = high, low
        lut, legend = cells.numeric_lut(labels, values, colormap=colormap, low=low, high=high)
        if low is not None or high is not None:
            import numpy as np
            import pandas as pd

            numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
            outside = ~np.isfinite(numbers)
            if low is not None:
                outside |= numbers < low
            if high is not None:
                outside |= numbers > high
            lut[np.asarray(labels)[outside]] = 0
            legend = {**legend, "shown": int((~outside).sum())}
        legend = {"kind": "numeric", **legend}
    else:
        saved = layer.get("colors") or {}
        import pandas as pd

        categories = pd.Series(values).dropna().astype(str).unique()
        existing = {c: saved[_sanitize(c)] for c in categories if _sanitize(c) in saved}
        from spacemap.phenotype_presets import ensure
        existing.update(ensure(categories))
        lut, items = cells.categorical_lut(labels, values, colors=existing, hidden=set(hidden))
        legend = {"kind": "categorical", "items": items}
    if len(_LUTS) > 64:
        _LUTS.clear()
    _LUTS[key] = (lut, legend)
    return lut, legend


def _numbers(values) -> bool:
    """Whether a column read from a table (its values come as objects) is numbers with more than 24 values, so it is
    shown as a colour scale (a marker, a distance) rather than as groups."""
    import pandas as pd

    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series) or series.dropna().map(lambda v: isinstance(v, str)).any():
        return False
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.notna().sum() == series.notna().sum() and numbers.nunique(dropna=True) > 24


def _limits(params):
    """The range of a numeric colour (lo, hi), each None when left empty."""
    import math

    def number(text):
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    return number(params.get("lo")), number(params.get("hi"))


# Cell outlines on their own threads: the first outline of a newly opened ROI no longer waits behind the legend,
# histogram and image tile requests sharing the server's pool (K6).
_OUTLINE_THREADS = None


async def cell_tile(request: Request) -> Response:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    global _OUTLINE_THREADS
    if _OUTLINE_THREADS is None:
        _OUTLINE_THREADS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="outlines")
    return await asyncio.get_running_loop().run_in_executor(_OUTLINE_THREADS, _cell_tile, request)


def _cell_tile(request: Request) -> Response:
    from spacemap.viewer import cells

    layer, image, error = _layer(request, "cells")
    if error:
        return error
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    params = request.query_params
    hidden = frozenset(h for h in params.get("hide", "").split("\x1f") if h)
    lut = None
    if params.get("col"):
        lut, _ = _cell_lut(
            layer, params.get("t", ""), params["col"], hidden, params.get("cmap", "viridis"), _limits(params)
        )
    outline = params.get("ol", "ffff00")
    rgba_outline = tuple(int(outline[i : i + 2], 16) for i in (0, 2, 4)) + (255,)
    if not cells.pyramid_ready(layer["pyramid"]):
        import numpy as np

        empty = _png(np.zeros((1, 1, 4), dtype=np.uint8))  # outlines still being prepared
        empty.headers["Cache-Control"] = "no-store"
        return empty
    labels = cells.label_tile(layer["pyramid"], image, z, x, y)
    if params.get("f"):  # the layer's filters: cells not passing them are not drawn at all
        from spacemap_gui.studio import cell_view

        try:
            table = (layer.get("tables") or {}).get(params.get("t", ""))
            lookup, _, _ = cell_view.keep(table, layer["roi"], cell_view.parse_filters(params["f"]),
                                          roi_column=layer.get("roi_column", "batch_id"), label_column=layer.get("label_column", "label"))
        except (ValueError, OSError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if lookup is not None:
            import numpy as np

            inside = labels < len(lookup)
            shown = np.zeros(labels.shape, bool)
            shown[inside] = lookup[labels[inside]]
            labels = np.where(shown, labels, 0)
    if params.get("selection"):
        from spacemap_gui.studio.selection import tile_mask

        try:
            labels = tile_mask(
                params["selection"], (layer.get("tables") or {}).get(params.get("t")), layer["roi"], labels
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    return _png(cells.render_cell_tile(labels, lut, mode=params.get("mode", "outline"), outline_rgba=rgba_outline, z=z))


def cell_ready(request: Request) -> Response:
    """Whether the ROI's cell outlines are ready, and how many cells they hold."""
    from spacemap.viewer import cells

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    ready = cells.pyramid_ready(layer["pyramid"])
    return JSONResponse({"ready": ready, "cells": cells.pyramid_cell_count(layer["pyramid"]) if ready else None})


def cell_legend(request: Request) -> Response:
    layer, _, error = _layer(request, "cells")
    if error:
        return error
    params = request.query_params
    hidden = frozenset(h for h in params.get("hide", "").split("\x1f") if h)
    try:
        _, legend = _cell_lut(
            layer, params.get("t", ""), params.get("col", ""), hidden, params.get("cmap", "viridis"), _limits(params)
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(legend or {})


def cell_columns(request: Request) -> Response:
    from spacemap import dataio

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    table = (layer.get("tables") or {}).get(request.query_params.get("t", ""))
    if not table:
        return JSONResponse({"columns": []})
    from spacemap_gui.studio import cell_view

    try:
        kinds = cell_view.column_kinds(table)
    except (OSError, ValueError):
        kinds = {}
    return JSONResponse({"columns": dataio.columns(table), "kinds": kinds})


def cell_filter_count(request: Request) -> Response:
    """How many of the image's cells pass the layer's filters, and a filter column's values (to pick from)."""
    from spacemap.viewer import cells
    from spacemap_gui.studio import cell_view

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    params = request.query_params
    table = (layer.get("tables") or {}).get(params.get("t", ""))
    if not table:
        return JSONResponse({"error": "Choose the cells' data first."}, status_code=400)
    keys = {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}
    try:
        found = {}
        if params.get("values"):  # a column's values: groups with their counts, or the range of numbers
            _, values = cells.roi_values(table, layer["roi"], params["values"], **keys)
            import pandas as pd

            series = pd.Series(values)
            numbers = pd.to_numeric(series, errors="coerce")
            if cell_view.column_kinds(table).get(params["values"]) == "value" and numbers.notna().any():
                found["range"] = [float(numbers.min()), float(numbers.max())]
            else:
                found["groups"] = [[str(k), int(v)] for k, v in series.dropna().astype(str).value_counts().head(500).items()]
        _, shown, total = cell_view.keep(table, layer["roi"], cell_view.parse_filters(params.get("f", "")), **keys)
        found.update(shown=shown, total=total)
        return JSONResponse(found)
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def cell_at(request: Request) -> Response:
    from spacemap.viewer import cells

    layer, image, error = _layer(request, "cells")
    if error:
        return error
    try:
        if "label" in request.query_params:
            label = int(request.query_params["label"])
            if label <= 0:
                raise ValueError("Invalid cell number")
        else:
            x, y = float(request.query_params["x"]), float(request.query_params["y"])
            label = _label_at(layer, x, y)
    except (KeyError, ValueError):
        return JSONResponse({"error": "x and y are required"}, status_code=400)
    result = {"label": label}
    table = (layer.get("tables") or {}).get(request.query_params.get("t", ""))
    keys = {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}
    if label and table:
        try:
            result["details"] = cells.cell_details(table, layer["roi"], label, **keys)
        except Exception as exc:
            result["details_error"] = str(exc)
        if request.query_params.get("details_only") != "1":
            try:
                result["ranking"] = cells.marker_ranking(table, layer["roi"], label, _markers(layer, table), **keys)
            except Exception as exc:
                result["ranking_error"] = str(exc)
    return JSONResponse(result)


def cell_outline(request: Request) -> Response:
    """The clicked cell's boundary, to highlight it (x, y: the click; else the cell's centroid in the table)."""
    from spacemap.viewer import cells

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    params = request.query_params
    try:
        label = int(params["label"])
        if label <= 0:
            raise ValueError
    except (KeyError, ValueError):
        return JSONResponse({"error": "Give the cell's label."}, status_code=400)
    if not cells.pyramid_ready(layer["pyramid"]):
        return JSONResponse({"label": label, "rings": []})
    try:
        x, y = float(params["x"]), float(params["y"])
    except (KeyError, ValueError):
        import numpy as np

        table = (layer.get("tables") or {}).get(params.get("t", ""))
        columns = cells.centroid_columns(table) if table else None
        if not columns:
            return JSONResponse({"label": label, "rings": []})
        keys = {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}
        labels, xs = cells.roi_values(table, layer["roi"], columns[0], **keys)
        _, ys = cells.roi_values(table, layer["roi"], columns[1], **keys)
        found = np.flatnonzero(labels == label)
        if not found.size:
            return JSONResponse({"label": label, "rings": []})
        x, y = float(xs[found[0]]), float(ys[found[0]])
    return JSONResponse({"label": label, "rings": cells.cell_outline(layer["pyramid"], label, x, y)})


async def cell_profile(request: Request) -> Response:
    """Measure a clicked cell only when requested; never during image opening."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import cell_profiles, cells, marker_transforms

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    try:
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected cell measurement settings.")
        from spacemap_gui.studio.cell_links import table_layer
        layer = table_layer(layer, str(body.get("t", "")))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    def work():
        requested_label = body.get("label", 0)
        label = int(requested_label)
        if isinstance(requested_label, bool) or label <= 0 or label > 2**32 - 1 or float(requested_label) != label:
            raise ValueError("Choose a numbered cell first.")
        source = body.get("source", "table")
        selected_steps = marker_transforms.steps(body.get("transforms", []))
        if source == "table":
            table = (layer.get("tables") or {}).get(str(body.get("t", "")))
            if not table:
                raise ValueError("Choose a cell table first.")
            record = cells.cell_details(
                table,
                layer["roi"],
                label,
                roi_column=layer.get("roi_column", "batch_id"),
                label_column=layer.get("label_column", "label"),
            )
            import math

            markers = _markers(layer, table, include_nuclear=True)
            if record.get("_note") and Path(table).stat().st_size <= 16 * 1024 * 1024:
                # Small CSVs intentionally have no fast copy. Read their row
                # directly; retain any pending values already in the record.
                import pandas as pd

                positions, labels = cells.roi_index(
                    table,
                    layer["roi"],
                    roi_column=layer.get("roi_column", "batch_id"),
                    label_column=layer.get("label_column", "label"),
                )
                rows = positions[labels == label]
                if rows.size and markers:
                    values = pd.read_csv(table, usecols=markers, nrows=int(rows[0]) + 1).iloc[-1]
                    for marker in markers:
                        record.setdefault(marker, values[marker])
                    record.pop("_note", None)
            measurements = []
            for marker in markers:
                try:
                    value = float(record[marker])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    measurements.append({"marker": marker, "mean": value})
            profile = {
                "status": "ready",
                "source": "table",
                "table": Path(table).name,
                "label": label,
                "markers": measurements,
                "message": record.get("_note", ""),
            }
            settings = {
                "table": table,
                "roi": layer["roi"],
                "markers": markers,
                "roi_column": layer.get("roi_column", "batch_id"),
                "label_column": layer.get("label_column", "label"),
            }
            return marker_transforms.apply_profile(profile, selected_steps, settings)
        if source not in ("image", "corrected"):
            raise ValueError("Choose saved measurements or image pixels.")
        corrected = source == "corrected"
        image = layer.get("background_image") if corrected else layer["image"]
        if not image:
            raise ValueError("No matching background-subtracted image was found for this ROI.")
        if "_bg_subtracted.ome." in Path(image).name.lower():
            corrected = True
        segmentation = layer.get("segmentation") or cells._recorded_source(Path(layer["pyramid"])).get("source")
        if not segmentation:
            raise ValueError("Reopen this image to connect its cell outlines.")
        backgrounds = body.get("backgrounds") or {}
        if not isinstance(backgrounds, dict):
            raise ValueError("Background levels must be marker names and numbers.")
        if "Z1" in selected_steps:
            raise ValueError("Z1 compares cells in this ROI. Choose Saved measurements to use it.")
        profile = cell_profiles.image_profile(
            image,
            layer["pyramid"],
            segmentation,
            label,
            backgrounds,
            corrected=corrected,
            retry=body.get("retry") is True,
        )
        return marker_transforms.apply_profile(profile, selected_steps)

    try:
        return JSONResponse(await run_in_threadpool(work))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


_MARKERS: dict = {}


def _markers(layer: dict, table: str, *, include_nuclear=False) -> list[str]:
    """The image's markers that the table has as columns (raw intensities). Cached."""
    from spacemap import dataio
    from spacemap.viewer.sources import fingerprint

    key = (layer["image"], fingerprint(layer["image"]), table, fingerprint(table), include_nuclear)
    if key not in _MARKERS:
        columns = set(dataio.columns(table))
        names = image_info(layer["image"]).channel_names
        excluded = ("HE_",) if include_nuclear else ("HE_", "DAPI")
        _MARKERS[key] = [n for n in names if n in columns and not n.upper().startswith(excluded)]
    return _MARKERS[key]


def warm_markers(layer: dict) -> None:
    """Read a ROI's marker values ahead of the first click (run in the background)."""
    from spacemap.viewer import cells

    for table in list((layer.get("tables") or {}).values())[:1]:
        try:
            cells.roi_matrix(
                table,
                layer["roi"],
                _markers(layer, table),
                roi_column=layer.get("roi_column", "batch_id"),
                label_column=layer.get("label_column", "label"),
            )
        except Exception:
            pass


def _label_at(layer: dict, x: float, y: float) -> int:
    from spacemap.viewer import cells

    if not cells.pyramid_ready(layer["pyramid"]):
        return 0
    source = open_source(layer["pyramid"])
    info = source.info
    if not (0 <= x < info.width and 0 <= y < info.height):
        return 0
    pixel = source.read_region(0, 0, int(x), int(y), int(x) + 1, int(y) + 1)
    return int(pixel[0, 0]) if pixel.size else 0


def _labels_in_rect(layer: dict, x0: float, y0: float, x1: float, y1: float) -> list[int]:
    import numpy as np

    from spacemap.viewer import cells

    if not cells.pyramid_ready(layer["pyramid"]):
        raise ValueError("The cell outlines are still being prepared; try again in a moment.")
    source = open_source(layer["pyramid"])
    info = source.info
    x0, x1 = sorted((max(0, int(x0)), min(info.width, int(x1))))
    y0, y1 = sorted((max(0, int(y0)), min(info.height, int(y1))))
    if (x1 - x0) * (y1 - y0) > 8192 * 8192:
        raise ValueError("Box is too large; zoom in and draw a smaller one.")
    region = source.read_region(0, 0, x0, y0, x1, y1)
    return [int(v) for v in np.unique(region) if v > 0]


async def cell_assign(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import cells

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Expected assignment settings.")
    except Exception:
        return JSONResponse({"error": "Expected JSON."}, status_code=400)
    from spacemap_gui.studio.cell_links import table_layer
    layer = table_layer(layer, str(body.get("t", "")))
    table = (layer.get("tables") or {}).get(str(body.get("t", "")))
    column = str(body.get("column") or "").strip()
    if not table or not column:
        return JSONResponse({"error": "Choose a table and a column to label."}, status_code=400)
    keys = {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}

    def work():
        if isinstance(body.get("values"), dict) and not body.get("request_id"):  # older clients
            return {"assigned": cells.assign_many(table, layer["roi"], body["values"], column, **keys), "previous": {}}
        labels = body.get("labels") or []
        if body.get("points"):
            labels = [_label_at(layer, float(x), float(y)) for x, y in body["points"]]
        if body.get("rect"):
            labels = _labels_in_rect(layer, *body["rect"])
        if body.get("request_id"):
            from spacemap.viewer import reference

            values = (
                body.get("values")
                if isinstance(body.get("values"), dict)
                else {str(v): body.get("value", "") for v in labels if v > 0}
            )
            return reference.save(
                table,
                layer["roi"],
                column,
                values,
                body["request_id"],
                source=body.get("source", "typed"),
                expected=body.get("expected"),
                provenance=body.get("provenance"),
                **keys,
            )
        return cells.assign(table, layer["roi"], labels, column, body.get("value", ""), **keys)

    try:
        result = await run_in_threadpool(work)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not save: {exc}"}, status_code=500)
    _LUTS.clear()
    return JSONResponse(result)


async def cell_suggestions(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap.viewer import consensus_model, suggestions

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    try:
        body = await _json_body(request)
        from spacemap_gui.studio.cell_links import table_layer
        layer = table_layer(layer, str(body.get("t", "")))
        table = (layer.get("tables") or {}).get(str(body.get("t", "")))
        if not table:
            raise ValueError("Choose a cell table first.")
        label = body.get("label")
        scan = body.get("scan") is True
        saved_model = body.get("mode") == "saved_reference"
        forest = body.get("mode") == "consensus_rf" or saved_model
        prepare = forest and body.get("prepare") is True
        if (
            not scan
            and not prepare
            and (isinstance(label, bool) or label is None or not 0 < int(label) < 2**32 or float(label) != int(label))
        ):
            raise ValueError("Choose a numbered cell first.")
        exclude = body.get("exclude") or []
        if not isinstance(exclude, list) or len(exclude) > 10000:
            raise ValueError("Reopen the image to start a fresh review list.")

        def work():
            if forest:
                if saved_model:
                    from spacemap_gui.studio import reference_models

                    settings = {
                        "table": table,
                        "roi": layer["roi"],
                        "destination": body.get("destination") or "phenotype_ref1",
                        "roi_column": layer.get("roi_column", "batch_id"),
                        "label_column": layer.get("label_column", "label"),
                    }
                    return reference_models.request(
                        settings,
                        body.get("model_id"),
                        label=int(label) if label is not None else None,
                        prepare=prepare,
                        scan=scan,
                        exclude=[int(v) for v in exclude],
                        retry=body.get("retry") is True,
                        cancel=body.get("cancel") is True,
                    )
                settings = consensus_model.settings_for(
                    layer, table, _markers(layer, table, include_nuclear=True), body.get("destination")
                )
                return consensus_model.request(
                    settings,
                    label=int(label) if label is not None else None,
                    prepare=prepare,
                    scan=scan,
                    exclude=[int(v) for v in exclude],
                    retry=body.get("retry") is True,
                    cancel=body.get("cancel") is True,
                )
            settings = suggestions.settings_for(layer, table, body, _markers(layer, table, include_nuclear=True))
            return suggestions.request(
                settings,
                label=int(label) if label is not None else None,
                scan=scan,
                exclude=[int(v) for v in exclude],
                retry=body.get("retry") is True,
                cancel=body.get("cancel") is True,
            )

        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError, AttributeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not prepare suggestions: {exc}"}, status_code=500)


async def cell_reference_options(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap.viewer import reference

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    tables = layer.get("tables") or {}
    selected = tables.get(request.query_params.get("t", ""))
    if not selected:
        return JSONResponse({"columns": [], "phenotypes": [], "sources": []})
    try:
        return JSONResponse(await run_in_threadpool(reference.options, list(tables.values()), selected))
    except Exception as exc:
        return JSONResponse({"error": f"Could not read phenotype names: {exc}"}, status_code=400)


def cell_reference_models(request: Request) -> Response:
    from spacemap_gui.studio.reference_models import available

    layer = registry.resolve_object(request.path_params["token"], "cells") or {}
    table = (layer.get("tables") or {}).get(request.query_params.get("t", ""))
    if not table:
        return JSONResponse({"error": "Choose a registered cell table."}, status_code=400)
    return JSONResponse({"table":str(table), "models": available(table), "library": _library_models(table)})


def _library_models(table):
    """The model library's phenotyping models this table can use (at least one of their markers) and this project
    does not have yet: picking one imports it."""
    from spacemap import dataio
    from spacemap_gui.studio import model_library, reference_models

    try:
        columns = set(dataio.columns(table))
        found = []
        usable = {item["model_id"] for item in reference_models.available(table)}
        for item in model_library.entries():
            if item.get("kind") != "phenotyping" or item["id"] in usable:
                continue
            details = item.get("details") or {}
            markers = details.get("markers") or (item.get("record") or {}).get("markers") or []
            have = [m for m in markers if m in columns]
            record = item.get("record") or {}
            if not have or (len(have)<len(markers) and not ((details.get("adaptable") or record.get("adaptable")) and record.get("allow_missing", True))):
                continue
            found.append({"entry": item["entry"], "model_id": item["id"], "name": item.get("name", item["id"]),
                          "project": item.get("source_project", ""), "markers": len(markers), "present": len(have),
                          "accuracy": details.get("held_out_balanced_accuracy")})
        return found
    except (OSError, ValueError, KeyError):
        return []


async def cell_review(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap.viewer import reference

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    try:
        params = await _json_body(request) if request.method == "POST" else request.query_params
        if not hasattr(params, "get"):
            raise ValueError("Expected review settings.")
        from spacemap_gui.studio.cell_links import table_layer
        layer = table_layer(layer, str(params.get("t", "")))
        table = (layer.get("tables") or {}).get(str(params.get("t", "")))
        column = str(params.get("column") or "").strip()
        if not table or not column:
            raise ValueError("Choose a table and a reference column first.")
        keys = {"roi_column": layer.get("roi_column", "batch_id"), "label_column": layer.get("label_column", "label")}

        def work():
            if request.method == "POST":
                if not isinstance(params.get("flagged"), bool):
                    raise ValueError("Choose whether the cell needs review.")
                return reference.set_flag(table, layer["roi"], column, int(params["label"]), params["flagged"], **keys)
            if params.get("label"):
                return reference.get_flag(table, layer["roi"], column, int(params["label"]))
            return reference.review(
                table,
                layer["roi"],
                column,
                mode=params.get("mode", "recent"),
                offset=int(params.get("offset", 0)),
                **keys,
            )

        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _density_params(request: Request, layer: dict) -> tuple[str, dict]:
    params = request.query_params
    table = (layer.get("tables") or {}).get(params.get("t", ""))
    if not table:
        raise ValueError("Choose a table for the density map.")
    options = {
        "filter_column": params.get("fcol", ""),
        "filter_values": tuple(v for v in params.get("fvals", "").split("\x1f") if v),
        "value_column": params.get("vcol", ""),
        "radius_um": max(1.0, min(2000.0, float(params.get("r", 50)))),
        "weighted": params.get("w", "0") == "1",
        "roi_column": layer.get("roi_column", "batch_id"),
        "label_column": layer.get("label_column", "label"),
    }
    return table, options


def density_tile(request: Request) -> Response:
    from spacemap.viewer import cells

    layer, image, error = _layer(request, "cells")
    if error:
        return error
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    try:
        table, options = _density_params(request, layer)
        density = cells.density_grid(table, layer["roi"], image, **options)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return _png(
        cells.render_density_tile(density, image, z, x, y, colormap=request.query_params.get("cmap", "inferno"))
    )


def density_legend(request: Request) -> Response:
    from spacemap.viewer import cells

    layer, image, error = _layer(request, "cells")
    if error:
        return error
    try:
        table, options = _density_params(request, layer)
        density = cells.density_grid(table, layer["roi"], image, **options)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "kind": "numeric",
            "min": density["min"],
            "max": density["max"],
            "units": density["units"],
            "cells": density["cells"],
            "colormap": request.query_params.get("cmap", "inferno"),
        }
    )


def density_groups(request: Request) -> Response:
    """The groups of a column in this ROI (phenotypes, triad types) with counts and colours."""
    from spacemap.acc_utils import build_color_mapping
    from spacemap.viewer import cells

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    params = request.query_params
    table = (layer.get("tables") or {}).get(params.get("t", ""))
    column = params.get("col", "")
    if not table or not column:
        return JSONResponse({"items": []})
    try:
        counts = cells.group_counts(
            table,
            layer["roi"],
            column,
            roi_column=layer.get("roi_column", "batch_id"),
            label_column=layer.get("label_column", "label"),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if len(counts) > 200:
        return JSONResponse({"kind": "numeric", "items": []})
    saved = layer.get("colors") or {}
    names = [value for value, _ in counts]
    existing = {name: saved[_sanitize(name)] for name in names if _sanitize(name) in saved}
    from spacemap.phenotype_presets import ensure
    colours = build_color_mapping(sorted(names, key=str.casefold), existing={**existing, **ensure(names)})
    return JSONResponse(
        {
            "kind": "categorical",
            "items": [
                {"value": value, "count": count, "color": "#" + str(colours.get(value, "#888888")).lstrip("#")}
                for value, count in counts
            ],
        }
    )


def cell_stamp(request: Request) -> Response:
    """Changes whenever the table, or a column or label saved beside it, changes."""
    from spacemap.viewer import cells
    from spacemap.viewer.sources import fingerprint

    layer, _, error = _layer(request, "cells")
    if error:
        return error
    table = (layer.get("tables") or {}).get(request.query_params.get("t", ""))
    if not table:
        return JSONResponse({"stamp": ""})
    try:
        stamp = f"{fingerprint(table)}|{cells._edits_token(table)}"
    except OSError:
        stamp = ""
    return JSONResponse({"stamp": stamp})


def annotation_tile(request: Request) -> Response:
    from spacemap.viewer import annotations

    layer, image, error = _layer(request, "annotations")
    if error:
        return error
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    hidden = {h for h in request.query_params.get("hide", "").split("\x1f") if h}
    try:
        shapes = annotations.load_annotations(layer["path"])
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": f"Could not read the annotation file: {exc}"}, status_code=400)
    return _png(annotations.render_annotation_tile(shapes, image, z, x, y, hidden=hidden))


def annotation_legend(request: Request) -> Response:
    from spacemap.viewer import annotations

    layer, _, error = _layer(request, "annotations")
    if error:
        return error
    try:
        items = annotations.load_annotations(layer["path"]).legend()
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": f"Could not read the annotation file: {exc}"}, status_code=400)
    return JSONResponse({"kind": "categorical", "items": items})


def annotation_at(request: Request) -> Response:
    from spacemap.viewer import annotations

    layer, _, error = _layer(request, "annotations")
    if error:
        return error
    try:
        hidden = set(request.query_params.get("hide", "").split("\x1f"))
        return JSONResponse(annotations.annotation_at(layer["path"], float(request.query_params["x"]),
                                                      float(request.query_params["y"]), hidden))
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def annotation_profile(request: Request) -> Response:
    """Measure a clicked imported polygon using its original, current file geometry."""
    from starlette.concurrency import run_in_threadpool
    from shapely.geometry import mapping
    from spacemap.viewer import annotations
    from spacemap_gui.studio import regions

    layer, _, error = _layer(request, "annotations")
    if error:
        return error
    try:
        body = await _json_body(request)
        loaded = await run_in_threadpool(annotations.load_annotations, layer["path"])
        if not isinstance(body, dict) or body.get("stamp") != loaded.fingerprint:
            raise ValueError("The annotation file changed. Click the polygon again.")
        index = body.get("index")
        if type(index) is not int or not 0 <= index < len(loaded.geometries):
            raise ValueError("Choose a polygon in this annotation layer.")
        return JSONResponse(await run_in_threadpool(regions.profile, layer["image"],
            mapping(loaded.geometries[index]), layer.get("reference") or None))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# ---------------------------------------------------------------------------
# Shapes drawn in the viewer, and marker display settings
# ---------------------------------------------------------------------------

_MAX_BODY = 64 * 1024 * 1024


async def _json_body(request: Request):
    if int(request.headers.get("content-length") or 0) > _MAX_BODY:
        raise ValueError("That is too large to send (over 64 MB).")
    try:
        return await request.json()
    except Exception as exc:
        raise ValueError("Expected JSON.") from exc


def _shapes_or_404(request: Request):
    layer = registry.resolve_object(request.path_params["token"], "shapes")
    if layer is None:
        return None, JSONResponse({"error": "Unknown shapes layer. Reopen the viewer."}, status_code=404)
    return layer, None


def shapes_get(request: Request) -> Response:
    layer, error = _shapes_or_404(request)
    if error:
        return error
    try:
        if layer.get("annotation_kind"):
            from spacemap_gui.studio import annotation_edits

            landed = request.query_params.get("landed")
            if landed:  # did the page's save that timed out reach the file?
                return JSONResponse({"landed": annotation_edits.last_save_id(layer["path"]) == landed,
                                     "revision": annotation_edits.revision(layer["path"])})
            if request.query_params.get("revision") == annotation_edits.revision(layer["path"]):
                return JSONResponse({"unchanged": True})
            features, version = annotation_edits.read(layer["path"], kind=layer["annotation_kind"], label=layer.get("tissue_label", "tissue"))
            return JSONResponse({**annotation_edits.display(features, version), "editable": bool(layer.get("editable", True)),
                                 "annotation_kind": layer["annotation_kind"], "regions": True, "saved_file": str(Path(layer["path"]).resolve())})
        from spacemap_gui.studio import annotation_edits
        features, version = annotation_edits.read(layer["path"], kind="shapes")
    except Exception as exc:
        return JSONResponse({"error": f"Could not read the shapes: {exc}"}, status_code=500)
    return JSONResponse({**annotation_edits.display(features, version), "editable": bool(layer.get("editable", True)),
                         "saved_file": str(Path(layer["path"]).resolve())})


async def shapes_save(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import shapes

    layer, error = _shapes_or_404(request)
    if error:
        return error
    if not layer.get("editable", True):
        return JSONResponse({"error": "These shapes are read-only here."}, status_code=403)
    try:
        body = await _json_body(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    def work():
        if layer.get("annotation_kind") or body.get("revision") is not None:
            from spacemap_gui.studio import annotation_edits

            import re

            ids = [value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{8,64}", value) else None
                   for value in (body.get("save_id"), body.get("after_save"))]
            return annotation_edits.save(layer["path"], body.get("features"), body.get("revision"),
                                         kind=layer.get("annotation_kind") or "shapes", label=layer.get("tissue_label", "tissue"),
                                         save_id=ids[0], after_save=ids[1])
        merged = shapes.merge_edits(shapes.load(layer["path"]), body.get("features") or [])
        return shapes.save(layer["path"], merged)

    try:
        saved = await run_in_threadpool(work)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except Exception as exc:
        return JSONResponse({"error": f"Could not save the shapes: {exc}"}, status_code=500)
    return JSONResponse(saved if isinstance(saved, dict) else {"count": len(saved)})


async def shapes_brush(request: Request) -> Response:
    """One brush stroke on the full saved polygon, never the simplified display copy."""
    from starlette.concurrency import run_in_threadpool
    from spacemap.viewer.sources import image_info
    from spacemap_gui.studio import annotation_edits

    layer, error = _shapes_or_404(request)
    if error:
        return error
    if not layer.get("editable", True):
        return JSONResponse({"error": "These shapes are read-only here."}, status_code=403)
    try:
        body = await _json_body(request)
        def work():
            info = image_info(layer["image"])
            return annotation_edits.brush(layer["path"], body.get("id"), body.get("points"), body.get("radius"), body.get("mode"),
                                           body.get("revision"), kind=layer.get("annotation_kind") or "shapes",
                                           label=layer.get("tissue_label", "tissue"), bounds=(info.width, info.height))
        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)


async def shapes_polygon_edit(request: Request) -> Response:
    """Split/merge saved annotations using full coordinates and revision checks."""
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import annotation_edits

    layer, error = _shapes_or_404(request)
    if error:
        return error
    if not layer.get("editable", True):
        return JSONResponse({"error": "These shapes are read-only here."}, status_code=403)
    try:
        body = await _json_body(request)
        if body.get("op") == "neck":
            return JSONResponse(await run_in_threadpool(annotation_edits.approve_neck, layer["path"], body.get("cutout"),
                                body.get("revision"), kind=layer.get("annotation_kind") or "shapes",
                                label=layer.get("tissue_label", "tissue"), ids=body.get('ids')))
        result = await run_in_threadpool(annotation_edits.edit, layer["path"], body.get("ids"), body.get("op"),
                                        body.get("revision"), line=body.get("line"), width=body.get("width", 2),
                                        gap=body.get("gap", 3), kind=layer.get("annotation_kind") or "shapes",
                                        label=layer.get("tissue_label", "tissue"))
        return JSONResponse(result)
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)


async def shapes_history(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import annotation_edits

    layer, error = _shapes_or_404(request)
    if error:
        return error
    if request.method == "GET":
        return JSONResponse(await run_in_threadpool(annotation_edits.history, layer["path"]))
    if not layer.get("editable", True):
        return JSONResponse({"error": "These shapes are read-only here."}, status_code=403)
    try:
        body = await _json_body(request)
        return JSONResponse(await run_in_threadpool(annotation_edits.restore, layer["path"], body.get("version"), body.get("revision"),
                                                    kind=layer.get("annotation_kind") or "shapes", label=layer.get("tissue_label", "tissue")))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)


async def shapes_edit(request: Request) -> Response:
    """Merge two shapes into one, or cut one along a line; the browser keeps the result (nothing is saved here)."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import shapes

    try:
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        if body.get("op") == "neck":
            from spacemap_gui.studio.annotation_edits import neck_features
            return JSONResponse(await run_in_threadpool(neck_features, body.get("features"), body.get("cutout")))
        if "geometries" in body:
            from spacemap_gui.studio.annotation_edits import edit_geometry
            return JSONResponse({"geometries": await run_in_threadpool(
                edit_geometry, body["geometries"], body.get("op"), line=body.get("line"),
                width=body.get("width", 2), gap=body.get("gap", 3))})
        if body.get("op") == "merge":
            parts = body.get("shapes")
            if not (isinstance(parts, list) and len(parts) == 2):
                raise ValueError("Merge needs two shapes.")
            found = await run_in_threadpool(shapes.join, parts[0], parts[1])
            return JSONResponse({"polygons": [found]})
        if body.get("op") == "split":
            return JSONResponse({"polygons": await run_in_threadpool(shapes.cut, body.get("shape"), body.get("line"))})
        if body.get("op") == "brush":
            from spacemap_gui.studio.annotation_edits import brush_geometry
            return JSONResponse({"polygons": await run_in_threadpool(brush_geometry, body.get("geometry"), body.get("points"), body.get("radius"), body.get("mode"))})
        raise ValueError("Unknown edit.")
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def shapes_profile(request: Request) -> Response:
    """The markers that stand out inside one region of a shapes layer, against the image's tissue."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import regions

    layer, error = _shapes_or_404(request)
    if error:
        return error
    try:
        body = await _json_body(request)
        if not isinstance(body, dict) or not isinstance(body.get("geometry"), dict):
            raise ValueError("Send the region's geometry.")
        return JSONResponse(await run_in_threadpool(regions.profile, layer["image"], body["geometry"], layer.get("reference") or None))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def shapes_sources(request: Request) -> Response:
    layer, error = _shapes_or_404(request)
    if error:
        return error
    return JSONResponse(
        {"sources": [{"id": i, "name": name} for i, (name, _) in enumerate(layer.get("sources") or [])]}
    )


async def shapes_import(request: Request) -> Response:
    """Add shapes from one of the project's files, or from features the browser read from a file."""
    import secrets

    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import shapes

    layer, error = _shapes_or_404(request)
    if error:
        return error
    if not layer.get("editable", True):
        return JSONResponse({"error": "These shapes are read-only here."}, status_code=403)
    try:
        body = await _json_body(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    def work():
        if "source" in body:
            sources = layer.get("sources") or []
            index = int(body["source"])
            if not 0 <= index < len(sources):
                raise ValueError("Unknown file.")
            incoming = shapes.load(sources[index][1])
        else:
            incoming = [f for f in (shapes.normalise(f, i) for i, f in enumerate(body.get("features") or [])) if f]
        for feature in incoming:  # new ids, so an import never replaces a shape already here
            feature["properties"]["id"] = secrets.token_hex(4)
        if body.get("revision") is not None or layer.get("annotation_kind"):
            from spacemap_gui.studio import annotation_edits
            current, _ = annotation_edits.read(layer["path"], kind=layer.get("annotation_kind") or "shapes", label=layer.get("tissue_label", "tissue"))
            result = annotation_edits.replace(layer["path"], current + incoming, body.get("revision"), "Imported polygons", edited=True,
                                               kind=layer.get("annotation_kind") or "shapes", label=layer.get("tissue_label", "tissue"))
            return {**result, "added": len(incoming), "count": len(result["features"]), "complete": True}
        saved = shapes.save(layer["path"], shapes.load(layer["path"]) + incoming)
        return {"added": len(incoming), "count": len(saved), "features": shapes.for_display(incoming)}

    try:
        result = await run_in_threadpool(work)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not import: {exc}"}, status_code=500)
    return JSONResponse(result)


def shapes_download(request: Request) -> Response:
    from pathlib import Path

    layer, error = _shapes_or_404(request)
    if error:
        return error
    path = Path(layer["path"])
    body = path.read_bytes() if path.is_file() else b'{"type": "FeatureCollection", "features": []}'
    return Response(
        body, media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="{path.name}"'}
    )


# ---------------------------------------------------------------------------
# GeoJSON files added to the view, and viewer windows
# ---------------------------------------------------------------------------


def _workspace_or_404(request: Request):
    workspace = registry.resolve_object(request.path_params["token"], "workspace")
    if workspace is None:
        return None, JSONResponse({"error": "Unknown project. Reopen the viewer."}, status_code=404)
    return workspace, None


def _annotation_layer(path: str, image: str, image_token: str, name: str) -> dict:
    return {
        "id": "file-" + registry.register_object("annotations", path=path, image=image)[:8],
        "type": "annotations",
        "name": name,
        "token": registry.register_object("annotations", path=path, image=image),
        "image_token": image_token,
        "opacity": 0.7,
        "visible": True,
    }


def workspace_files(request: Request) -> Response:
    """GeoJSON files in the project's data and outputs folders, newest first."""
    from pathlib import Path
    from spacemap.viewer.catalog import annotation_files

    workspace, error = _workspace_or_404(request)
    if error:
        return error
    root = Path(workspace["root"])
    found = []
    for folder in ("outputs", "data"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in annotation_files(base):
            if path.name.endswith("_cell_boundaries.geojson"):
                continue
            try:
                found.append((path.stat().st_mtime, str(path.relative_to(root))))
            except OSError:
                continue
            if len(found) >= 3000:
                break
    found.sort(reverse=True)
    return JSONResponse({"files": [relative for _, relative in found]})


async def workspace_view(request: Request) -> Response:
    """The last display of this registered image, stored with its project."""
    from pathlib import Path
    from .settings_io import run as settings_io
    from spacemap.viewer import view_state

    workspace, error = _workspace_or_404(request)
    if error:
        return error
    root, image = workspace['root'], workspace['image']

    def valid_extra(layer):
        kind = layer.get('type')
        if kind == 'image':
            return bool(registry.resolve(layer.get('token')))
        found = registry.resolve_object(layer.get('token'), 'cells' if kind == 'density' else kind)
        if kind == 'annotations' and found and not Path(found.get('path', '')).is_file():
            return False
        return bool(found and Path(found.get('image', '')).resolve() == Path(image).resolve())

    try:
        if request.method == 'GET':
            data = await settings_io(view_state.load, view_state.path_for(root, image))
            if data:
                data['extras'] = [layer for layer in data.get('extras', []) if valid_extra(layer)]
            return JSONResponse({'view': data})
        body = await _json_body(request)
        record = body.get('view')
        if isinstance(record, dict) and any(not valid_extra(layer) for layer in record.get('extras', []) if isinstance(layer, dict)):
            raise ValueError('An added layer is no longer registered for this image. Reopen it before saving the view.')
        data = await settings_io(view_state.save, root, image, record, body.get('stamp'))
        return JSONResponse({'saved_at': data['saved_at']})
    except (ValueError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
    except OSError as exc:
        return JSONResponse({'error': f'Could not access the saved viewer settings: {exc}'}, status_code=500)


async def workspace_add(request: Request) -> Response:
    """Show a GeoJSON file on the image: one of the project's, or one sent from the browser."""
    import json
    import re
    from pathlib import Path

    from starlette.concurrency import run_in_threadpool

    workspace, error = _workspace_or_404(request)
    if error:
        return error
    try:
        body = await _json_body(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    root = Path(workspace["root"]).resolve()

    def work():
        if body.get("relative"):
            path = (root / str(body["relative"])).resolve()
            if root not in path.parents or path.suffix.lower() not in (".geojson", ".json") or not path.is_file():
                raise ValueError("That file is not in this project.")
            name = str(body["relative"])
        else:
            data = body.get("geojson")
            if isinstance(data, str):
                data = json.loads(data)
            features = data.get("features") if isinstance(data, dict) else data
            if not isinstance(features, list) and not (isinstance(data, dict) and data.get("type") == "Feature"):
                raise ValueError("That file is not GeoJSON.")
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(body.get("name") or "imported")).stem)[:80] or "imported"
            folder = Path(workspace["imports"])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{stem}.geojson"
            path.write_text(json.dumps(data))
            name = f"Imported / {path.name}"
        from spacemap.viewer import annotations

        count = len(annotations.load_annotations(path).geometries)
        return str(path), name, count

    try:
        path, name, count = await run_in_threadpool(work)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not read the file: {exc}"}, status_code=500)
    layer = _annotation_layer(path, workspace["image"], registry.register(workspace["image"]), name)
    return JSONResponse({"layer": layer, "shapes": count})


def studio_commands(request: Request) -> Response:
    from spacemap_gui.studio.schemas import catalogue

    if registry.session_view(request.path_params["channel"], mark_seen=False) is None:
        return JSONResponse({"error": "Open Studio from SPACEMAP first."}, status_code=404)
    return JSONResponse({"commands": catalogue()})


async def studio_command(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    try:
        body = await _json_body(request)

        def work():
            from spacemap_gui.studio.commands import execute
            if not body.get("assistant"):
                return execute(request.path_params["channel"], body)
            from spacemap_gui.studio import chat, chat_actions
            checked, context = chat_actions.prepare_command(request.path_params["channel"], body, chat._store())
            result = execute(request.path_params["channel"], checked)
            chat_actions.record(chat._store(), context, result)
            return result

        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_cluster_names(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import cluster_names, jobs
    try:
        root = _studio_project(request)
        project = registry.session_project(request.path_params["channel"]) or {}
        def work():
            job = jobs.status(root, request.query_params.get("job_id", ""))
            result = job.get("result") or {}
            if job.get("status") != "done" or result.get("table") != project.get("unified") or not result.get("clusters"):
                raise ValueError("Choose a finished clustering run of the selected table.")
            return cluster_names.saved_names(root, result["table"], result["column"], cluster_job_id=job["id"])
        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)


async def studio_chat(request: Request) -> Response:
    """Studio's assistant: the conversation and the model's state (GET), or an answer to a message (POST)."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import chat

    channel = request.path_params["channel"]
    try:
        if registry.session_view(channel, mark_seen=False) is None:
            raise ValueError("Open Studio from SPACEMAP first.")
        if request.method == "GET":
            conversation = request.query_params.get("conversation", "")
            return JSONResponse(
                {
                    "messages": await run_in_threadpool(chat.history, conversation),
                    "model": await run_in_threadpool(chat.model_state),
                    "memories": await run_in_threadpool(chat.memories),
                }
            )
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        answer = (
            lambda on_text=None: chat.reply(  # noqa: E731
                channel,
                body.get("text"),
                conversation=body.get("conversation"),
                step=str(body.get("step") or "")[:16],
                images=body.get("images"),
                view=body.get("view"),
                page=body.get("page"),
                on_text=on_text,
                mode=body.get("mode", "auto"),
                cluster_review=body.get("cluster_review"),
            )
        )
        if body.get("stream"):
            return _stream_answer(answer)
        return JSONResponse(await run_in_threadpool(answer))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _stream_answer(answer):
    """The answer as it is written: one JSON line per piece ({"text": ...}), then {"done": reply} or {"error": ...}."""
    import asyncio
    import json
    import threading

    from starlette.concurrency import run_in_threadpool
    from starlette.responses import StreamingResponse

    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    stopped = threading.Event()
    def send(event):
        if stopped.is_set():
            raise RuntimeError("The chat request was stopped.")
        loop.call_soon_threadsafe(events.put_nowait, event)

    def work():
        try:
            send({"done": answer(lambda piece: send(piece if isinstance(piece, dict) else {"text": piece}))})
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            if not stopped.is_set():
                send({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the page waits for a last line, whatever went wrong
            if not stopped.is_set():
                send({"error": f"The assistant stopped: {exc}"})

    async def lines():
        task = asyncio.ensure_future(run_in_threadpool(work))
        try:
            yield json.dumps({"status":"Request received; preparing the review…"}) + "\n"
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=5)
                except asyncio.TimeoutError:
                    yield '{"heartbeat":true}\n'
                    continue
                yield json.dumps(event, default=str) + "\n"
                if "done" in event or "error" in event:
                    break
        finally:
            stopped.set()
            if not task.done():
                task.cancel()

    return StreamingResponse(
        lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Content-Encoding":"identity"}
    )


async def studio_memory(request: Request) -> Response:
    """Remember something for the assistant, or forget a memory; both are kept locally."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import chat

    try:
        if registry.session_view(request.path_params["channel"], mark_seen=False) is None:
            raise ValueError("Open Studio from SPACEMAP first.")
        body = await _json_body(request)
        if body.get("forget"):
            return JSONResponse({"memories": await run_in_threadpool(chat.forget, str(body["forget"]))})
        return JSONResponse({"memories": await run_in_threadpool(chat.remember, body.get("content"))})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_ask(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    try:
        body = await _json_body(request)

        def work():
            from spacemap_gui.studio.assistant import propose

            return propose(
                request.path_params["channel"], body.get("text"), use_model=body.get("use_model", True) is True
            )

        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"status": "question", "message": str(exc)})


async def studio_describe(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    try:
        body = await _json_body(request)

        def work():
            from spacemap_gui.studio.assistant import describe

            return describe(request.path_params["channel"], body.get("image"))

        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"status": "question", "message": str(exc)})


def studio_context(request: Request) -> Response:
    from spacemap_gui.studio.shell import context

    try:
        base = request.url.path.split(MOUNT_PATH, 1)[0] if MOUNT_PATH in request.url.path else ""
        return JSONResponse(context(request.path_params["channel"], base))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


def _studio_project(request, *, write=False):
    channel = request.path_params["channel"]
    if registry.session_view(channel, mark_seen=False) is None:
        raise ValueError("Open Studio from SPACEMAP first.")
    project = registry.session_project(channel) or {}
    if not project.get("root"):
        raise ValueError("Load a project first.")
    if write and project.get("read_only"):
        raise ValueError("This project is open for viewing only.")
    return project["root"]


def studio_step(request: Request) -> Response:
    from spacemap_gui.studio.steps import work_list

    try:
        return JSONResponse(work_list(request.path_params["channel"], request.path_params["step"]))
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_overview(request: Request) -> Response:
    from spacemap_gui.studio.steps import overview

    try:
        return JSONResponse(overview(request.path_params["channel"]))
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_leiden_settings(request: Request) -> Response:
    """Tab 7's Leiden settings (its leiden_params.json): read, or save the ones Studio shows."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import leiden
    from spacemap_gui.studio.settings import revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(leiden.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = leiden.values(root)
        return JSONResponse({"values": saved, "revision": revision(saved)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_ai(request: Request) -> Response:
    """AI setup in Studio: the assistant's state (GET) or one setup action (POST). Settings are the user's, not the project's."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import ai_setup

    try:
        if registry.session_view(request.path_params["channel"], mark_seen=False) is None:
            raise ValueError("Open Studio from SPACEMAP first.")
        if request.method == "GET":
            return JSONResponse(await run_in_threadpool(ai_setup.state))
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        return JSONResponse(await run_in_threadpool(ai_setup.act, request.path_params.get("action", ""), body))
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_cluster_suggestions(request: Request) -> Response:
    """Names for a finished clustering's groups: reference cells in each and the reference model's vote."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import cluster_names, jobs, markers
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        project = registry.session_project(request.path_params["channel"]) or {}
        table = project.get("unified")
        if not table:
            raise ValueError("No combined table yet.")
        columns = await run_in_threadpool(cluster_names.reference_columns, table)
        job_id = request.query_params.get("job_id", "")
        if not job_id:
            return JSONResponse({"reference_columns": columns})
        done = jobs.status(root, job_id)
        result = done.get("result") or {}
        if done.get("kind") not in {"leiden", "som", "vortex", "subphenotype"} or done.get("status") != "done":
            raise ValueError("Choose a finished clustering.")
        if result.get("table") != table:
            raise ValueError("This clustering was made on another combined table.")
        chosen = request.query_params.get("reference") or (columns[0] if columns else "")
        if chosen not in columns:
            raise ValueError("Label some cells in a reference column (Phenotyping > Manual reference) first.")
        suffix = markers.suffix_of(read_config(root))
        found = await run_in_threadpool(
            cluster_names.suggest, table, result["column"], chosen, result.get("markers") or [], suffix
        )
        return JSONResponse({"reference_columns": columns, **found})
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_figure(request: Request) -> Response:
    """A publication figure (heatmap or dot plot) of a column's groups, or of a heatmap on screen; never saved here."""
    import base64
    import re

    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import figures
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        options = body.get("options") if isinstance(body.get("options"), dict) else {}
        title = str(body.get("title") or "")[:200]
        if body.get("source") == "run":
            from spacemap_gui.studio.phenotyping_inputs import combined_table
            if body.get("kind", "heatmap") != "heatmap":
                raise ValueError("Saved run profiles support heatmaps. Choose Full combined table for a dot plot.")
            options["title"] = title
            data, kind = await run_in_threadpool(figures.run_figure, root,
                combined_table(request.path_params["channel"]), body.get("job_id"), body.get("names"), options)
        elif body.get("matrix"):
            matrix = body["matrix"]
            data, kind = await run_in_threadpool(
                figures.matrix_figure, matrix.get("rows"), matrix.get("columns"), matrix.get("values"), title, options
            )
        else:
            project = registry.session_project(request.path_params["channel"]) or {}
            if not project.get("unified"):
                raise ValueError("No combined table yet.")
            bc = ((read_config(root).get("tab_configurations") or {}).get("tab6_batch_correction") or {}).get(
                "bc_config"
            )
            options.setdefault(
                "cofactor", float((bc if isinstance(bc, dict) else {}).get("arcsinh_cofactor", 5.0) or 5.0)
            )
            options["title"] = title
            names = body.get("names") if isinstance(body.get("names"), dict) else None
            data, kind = await run_in_threadpool(
                figures.table_figure,
                project["unified"],
                str(body.get("column") or ""),
                [str(m) for m in body.get("markers") or []],
                str(body.get("kind") or "heatmap"),
                options,
                names,
            )
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", title or "figure").strip("_") or "figure"
        return JSONResponse(
            {"data": base64.b64encode(data).decode(), "type": kind, "name": stem + "." + options.get("format", "png")}
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_saved(request: Request) -> Response:
    """Jobs > Saved results: what waits beside the table, and writing it in only when asked."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import saved

    try:
        _studio_project(request, write=request.method == "POST")
        project = registry.session_project(request.path_params["channel"]) or {}
        if request.method == "GET":
            return JSONResponse(await run_in_threadpool(saved.state, project))
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        return JSONResponse(await run_in_threadpool(saved.act, project, request.path_params.get("action", ""), body))
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_marker_choices(request: Request) -> Response:
    """The combined table's markers by name (raw, corrected), Tab 6's suffix, the saved sets and a method's choice."""
    from spacemap import dataio
    from spacemap_gui.studio import clustering, leiden, markers
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        project = registry.session_project(request.path_params["channel"]) or {}
        if not project.get("unified"):
            raise ValueError("No combined table yet: run batch correction first.")
        method = request.query_params.get("method", "")
        if method == "leiden":
            current = leiden.values(root)["markers"]
        elif method == "resolve":
            from spacemap_gui.studio import resolve_setup
            current = resolve_setup.get(root, project["unified"])["values"]["markers"]
        elif method == "consensus":
            from spacemap_gui.studio import consensus

            current = consensus.values(root)["markers"]
        elif method == "mr_ml":
            from spacemap_gui.studio import mr_ml
            current = mr_ml.values(root)['markers']
        elif method == "subphenotyping":
            from spacemap_gui.studio import subphenotypes

            current = subphenotypes.values(root)["markers"]
        elif method in clustering.METHODS:
            current = clustering.values(root, method)["markers"]
        else:
            current = []  # a figure: nothing chosen yet
        from spacemap_gui.studio.phenotyping_inputs import column_roles
        columns = dataio.columns(project["unified"])
        roles = column_roles(root, project["unified"], columns)
        found = markers.choices(root, read_config(root), columns, current)
        if method == "resolve" and not current:
            found["source"] = "raw"
        identifiers = set(roles.values()) | {"slide", "slide_id", "roi", "roi_id", "patient", "arm", "timepoint", "response", "original_label"}
        found["markers"] = [item for item in found["markers"] if item["name"] not in identifiers]
        found["groups"] = list(dict.fromkeys([c for c in (roles["roi"], roles["slide"], *found["groups"]) if c]))
        return JSONResponse({**found, "roles": roles})
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_subpheno_settings(request: Request) -> Response:
    """Tab 7's sub-phenotyping settings (its subphenotyping_params.json): read, or save the ones Studio shows."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import subphenotypes
    from spacemap_gui.studio.settings import revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(subphenotypes.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = subphenotypes.values(root)
        return JSONResponse({"values": saved, "revision": revision(saved)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_phenotype_values(request: Request) -> Response:
    """The phenotypes of a phenotype column of the combined table, with their cell counts."""
    from starlette.concurrency import run_in_threadpool

    from spacemap import dataio
    from spacemap_gui.studio import subphenotypes

    try:
        _studio_project(request)
        project = registry.session_project(request.path_params["channel"]) or {}
        column = request.query_params.get("column", "")
        if not project.get("unified"):
            raise ValueError("No combined table yet.")
        if not column.startswith("phenotype_") or column not in dataio.columns(project["unified"]):
            raise ValueError("Choose a phenotype column of the combined table.")
        found = await run_in_threadpool(subphenotypes.phenotype_values, project["unified"], column)
        return JSONResponse({"column": column, "values": found})
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_mrml_settings(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import mr_ml, reference_models, phenotyping_inputs
    from spacemap_gui.studio.settings import revision
    from spacemap import dataio
    from spacemap.viewer import reference
    try:
        root = Path(_studio_project(request, write=request.method == 'POST'))
        table = phenotyping_inputs.combined_table(request.path_params['channel'])
        if request.method == 'POST':
            body = await _json_body(request)
            saved = await run_in_threadpool(mr_ml.save, root, body.get('changes'), body.get('revision'))
        else:
            saved = mr_ml.values(root)
        models = await run_in_threadpool(reference_models.available, table)
        references = await run_in_threadpool(reference.columns, table)
        return JSONResponse({'values': saved, 'revision': revision(saved), 'models': models, 'references': references,
                             'columns': dataio.columns(table), 'table': Path(table).name})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def studio_consensus_settings(request: Request) -> Response:
    """Tab 7's Consensus ML settings (its consensus_ml_params.json): read, or save the ones Studio shows."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import consensus
    from spacemap_gui.studio.settings import revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(consensus.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = consensus.values(root)
        resources = await run_in_threadpool(consensus.hardware)
        return JSONResponse({"values": saved, "revision": revision(saved), "hardware": resources})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_clustering_settings(request: Request) -> Response:
    """A Tab 7 clustering method's settings (classic's own file): read, or save the ones Studio shows."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import clustering
    from spacemap_gui.studio.settings import revision

    try:
        method = request.path_params["method"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(clustering.save, root, method, body.get("changes"), body.get("revision"))
        else:
            saved = clustering.values(root, method)
        return JSONResponse({"values": saved, "revision": revision(saved)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_neighbourhood_settings(request: Request) -> Response:
    """Tab 8's neighbourhood settings (its tab8_nb_* keys), and the phenotype columns to choose from."""
    from starlette.concurrency import run_in_threadpool

    from spacemap import dataio
    from spacemap_gui.studio import neighbourhoods
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(neighbourhoods.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = neighbourhoods.values(read_config(root))
        project = registry.session_project(request.path_params["channel"]) or {}
        columns = dataio.columns(project["unified"]) if project.get("unified") else []
        return JSONResponse(
            {
                "values": saved,
                "revision": revision(saved),
                "phenotypes": [name for name in columns if "phenotype" in name.lower()],
                "columns": columns,
            }
        )
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_step_folders(request: Request) -> Response:
    """A step's folders (where it reads and writes); POST {"label", "folder"} points one at another folder."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import folders

    try:
        step = request.path_params["step"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            return JSONResponse(await run_in_threadpool(folders.change, root, step, body.get("label"), body.get("folder")))
        return JSONResponse(await run_in_threadpool(folders.listing, root, step))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_tagging_settings(request: Request) -> Response:
    """Tab 5's cell tagging settings, the cell tables with their annotation files, the rules and the folders to choose from."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import tagging
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            state = await run_in_threadpool(tagging.save, root, body.get("changes"), body.get("revision"))
        else:
            state = await run_in_threadpool(tagging.values, root, await run_in_threadpool(read_config, root))
        found = await run_in_threadpool(tagging.sources, root, state)
        shown = {kind: tagging._shown(root, state[key]) for kind, (key, _) in tagging.FOLDERS.items()}
        return JSONResponse(
            {
                "values": state,
                "revision": revision(state),
                "folders": shown,
                "paths": {kind: str(state[key]) for kind, (key, _) in tagging.FOLDERS.items()},
                "found": found,
                "rules": tagging.rules(found, state),
                "columns_added": await run_in_threadpool(tagging.planned_columns, state, tagging.rules(found, state)),
                # The project's pixel size, to show edge distances in µm (each image's own is used when tagging).
                "microns": (await run_in_threadpool(read_config, root)).get("mpp_codex"),
                "tagged": tagging.outputs(state, found),
                "csv_folders": await run_in_threadpool(tagging.folders_with, root, ".csv"),
                "geojson_folders": await run_in_threadpool(tagging.folders_with, root, ".geojson"),
            }
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_tagging_preview(request: Request) -> Response:
    """Tag the open ROI's cells in memory (POST {"clear": true} drops it); cells can then be coloured by a new column."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import tagging
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        view = registry.session_view(request.path_params["channel"], mark_seen=False) or {}
        layers = (view.get("spec") or {}).get("layers", [])
        image = next((layer for layer in layers if layer.get("type") == "image"), None)
        cells_layer = next((layer for layer in layers if layer.get("type") == "cells" and image
                            and layer.get("image_token") == image.get("token")), None)
        registered = registry.resolve_object(cells_layer["token"], "cells") if cells_layer else None
        if not image or not registered:
            raise ValueError("Open an ROI with cell outlines first.")
        body = await _json_body(request)
        if body.get("clear"):
            tagging.clear_preview(registered.get("image"), registered.get("roi"))
            return JSONResponse({"cleared": True})
        state = await run_in_threadpool(tagging.values, root, await run_in_threadpool(read_config, root))
        bounds = body.get("bounds")
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 4 or not all(isinstance(v, (int, float)) for v in bounds)):
            raise ValueError("Send the area in view as x, y, width, height.")
        found = await run_in_threadpool(tagging.preview, root, state, registry.resolve(image["token"]), registered.get("roi"), bounds)
        return JSONResponse({**found, "prefix": tagging.PREFIX})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_interaction_settings(request: Request) -> Response:
    """Tab 8's cell-cell interaction settings (its tab8_cci_* keys), and the phenotype columns to choose from."""
    from starlette.concurrency import run_in_threadpool

    from spacemap import dataio
    from spacemap_gui.studio import interactions
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(interactions.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = interactions.values(read_config(root))
        project = registry.session_project(request.path_params["channel"]) or {}
        columns = dataio.columns(project["unified"]) if project.get("unified") else []
        return JSONResponse(
            {
                "values": saved,
                "revision": revision(saved),
                "phenotypes": [name for name in columns if "phenotype" in name.lower()],
                "columns": columns,
            }
        )
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_marker_regions(request: Request) -> Response:
    """The marker-region settings and the open image's markers; POST {values} saves the settings, POST {values, preview}
    draws the regions they find on the open image without saving them."""
    import time as clock

    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, commands, regions
    from spacemap_gui.studio.settings import read_config

    try:
        channel = request.path_params["channel"]
        body = await _json_body(request) if request.method == "POST" else {}
        root = Path(_studio_project(request, write=request.method == "POST" and not body.get("preview")))
        config = await run_in_threadpool(read_config, root)
        markers, image, microns = [], None, None
        view = registry.session_view(channel, mark_seen=False) or {}
        if view.get("spec"):
            try:
                _, layer, info = commands._view(channel, view["spec"], {}, [])
                markers, image = [] if info.rgb else list(info.channel_names), registry.resolve(layer["token"])
                microns = info.pixel_size_um
            except (ValueError, KeyError, TypeError):
                pass
        if body.get('image_token') and registry.resolve(body['image_token']) != image:
            raise ValueError('The image changed. Run Marker regions on the open image again.')
        if request.method == "POST" and not body.get("preview"):
            values = await run_in_threadpool(regions.save, root, body.get("values"))
        else:
            values = regions.check(body["values"]) if body.get("values") is not None else regions.settings(config)
        from spacemap.background_detection import THRESHOLD_METHODS

        found = {"values": values, "markers": markers, "scales": list(regions.SCALES), "cutoffs": list(regions.CUTOFFS),
                 "methods": list(THRESHOLD_METHODS), "rule_default": regions.RULE_DEFAULT,
                 'microns_per_pixel':microns or annotation.settings(config).get('microns_per_pixel')}
        if image:
            from spacemap_gui.studio import annotation_layers, annotation_inputs
            project = registry.session_project(channel) or {}
            mappings = {**project.get('mappings', {}), **config.get('directory_mappings', {})}
            output = annotation.all_folders(root, mappings)['output']
            saved_layers = annotation_layers.generated(root, image, output)
            found['layers'] = [annotation_layers.layer(root,image,output,path,project.get('read_only',False))
                               for path, entry in saved_layers.items() if entry['kind']=='regions']
            selection = annotation_inputs.selection(root,image,'regions')
            found['target'] = str(regions.target(output,image,values,selection).resolve())
            found['read_only'] = bool(project.get('read_only'))
        if body.get("preview"):
            if not image:
                raise ValueError("Open an image first.")
            project = registry.session_project(channel) or {}
            mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
            output = annotation.all_folders(root, mappings)["output"]
            started = clock.perf_counter()
            from spacemap_gui.studio import annotation_inputs
            scope = await run_in_threadpool(annotation_inputs.scope_path, root, image, output, "regions")
            levels = (await run_in_threadpool(regions.saved_levels, root, config, mappings, image)) if values["cutoff"] == "saved" else None
            pool = (await run_in_threadpool(regions.pool_for, root, mappings, image, values)) if values["cutoff"] == "groups" else None
            features, stats = await run_in_threadpool(lambda: regions.find(image, {**values, "within": "tissue"}, scope, levels=levels, pool=pool))
            found.update(stats, outlines=annotation.rings(features, limit=4000, include_holes=True), ms=round((clock.perf_counter() - started) * 1000))
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_bring(request: Request) -> Response:
    """Files brought in by hand. POST /drop?name= (the file as the body) keeps a dropped file; /path {path} opens a
    typed file or lists a typed folder ({folder, files}; then {path: folder, name}); /overlay {image_token} lays an
    image over the open one; /open {image_token} adds it to the image list to open it."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation_inputs, drops

    from . import ui

    try:
        channel, action = request.path_params["channel"], request.path_params["action"]
        root = Path(_studio_project(request, write=True))

        async def answer(path):
            if drops.kind_of(path.name) == "table":  # cell data: set up in the Cell data dialog
                return {"kind": "table", "name": path.name, "file_token": registry.register_object("datafile", path=str(path))}
            if drops.kind_of(path.name) == "geojson":
                features = await run_in_threadpool(drops.read_shapes, path)
                if drops.looks_like_cells(features):  # numbered cell outlines: offered as cells first
                    return {"kind": "cells", "name": path.name, "count": len(features),
                            "file_token": registry.register_object("datafile", path=str(path))}
                return {"kind": "geojson", "name": path.name, "features": features}
            image = await run_in_threadpool(drops.as_tiff, path)
            facts = await run_in_threadpool(drops.describe, image)
            return {"kind": "image", "image_token": registry.register(str(image)), **facts}

        if action == "drop":
            target = drops.target(root, request.query_params.get("name", ""))
            with target.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            return JSONResponse(await answer(target))
        body = await _json_body(request)
        if action == "shapes":  # cell outlines brought in, wanted as annotations after all
            source = (registry.resolve_object(str(body.get("file_token", "")), "datafile") or {}).get("path")
            if not source:
                raise ValueError("Bring the file in again.")
            return JSONResponse({"name": Path(source).name, "features": await run_in_threadpool(drops.read_shapes, Path(source))})
        if action == "path":
            typed = Path(str(body.get("path") or "").strip()).expanduser()
            if body.get("name"):
                typed = typed / Path(str(body["name"])).name
            if typed.is_dir():
                return JSONResponse({"folder": str(typed), "files": await run_in_threadpool(drops.listing, typed)})
            if not typed.is_file() or not drops.kind_of(typed.name):
                raise ValueError("Type the full path of a GeoJSON, an image or a folder holding them.")
            return JSONResponse(await answer(typed))
        image = registry.resolve(str(body.get("image_token") or ""))
        if not image:
            raise ValueError("Bring the image in again.")
        if action == "overlay":
            from spacemap.viewer.sources import image_info

            info = await run_in_threadpool(image_info, image)
            layer = ui.image_layer(image, name=Path(image).name, channels=ui.default_channels(list(info.channel_names)), opacity=0.6)
            layer.update(id="overlay-" + __import__("hashlib").sha256(image.encode()).hexdigest()[:12], added=True, overlay=True)
            return JSONResponse({"layer": layer})
        if action == "open":
            return JSONResponse(await run_in_threadpool(ui.window_add_image, channel, image))
        raise ValueError("Choose drop, path, overlay or open.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _open_entry(channel, token):
    """The image entry for an image token (from the session's list, else made from the image itself), and its place."""
    from spacemap.viewer.catalog import sample_of

    image = registry.resolve(token or "")
    if not image:
        raise ValueError("Open an image first.")
    entries = registry.session_images(channel) or []
    image_id = next((i for i, entry in enumerate(entries) if Path(entry["path"]).resolve() == Path(image).resolve()), None)
    if image_id is not None:
        return entries[image_id], image_id
    return {"group": "ROIs", "name": Path(image).name, "path": str(image), "sample": sample_of(Path(image).name)}, None


def _layers_of(channel, entry):
    """The layers the image opens with now (to hand one new layer to the page)."""
    from . import ui

    project = registry.session_project(channel) or {}
    with ui.using_context(project):
        layers, _ = ui.layers_for_image(entry, context=project.get("tab", ""))
    return layers


async def studio_phenotyping_overview(request: Request) -> Response:
    """The combined table Phenotyping works on: its cells per image and its phenotype columns."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import phenotyping_overview

    try:
        _studio_project(request)
        project = registry.session_project(request.path_params["channel"]) or {}
        if request.query_params.get("results"):  # Spatial analysis: also which images each neighbourhood run covered
            from spacemap_gui.studio import tls
            found = await run_in_threadpool(phenotyping_overview.overview, project.get("unified"))
            tls_run = await run_in_threadpool(tls.latest, _studio_project(request), project.get("unified"))
            return JSONResponse({**found, "results": await run_in_threadpool(phenotyping_overview.results, project.get("unified")),
                                 "tls": {"counts": tls_run["counts"], "measured": tls_run["measured"], "column": tls_run["column"], "job_id": tls_run["job_id"]} if tls_run else None})
        return JSONResponse(await run_in_threadpool(phenotyping_overview.overview, project.get("unified")))
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_phenotyping_inputs(request: Request) -> Response:
    """Phenotyping's input/output controls and per-ROI result coverage."""
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import phenotyping_inputs

    try:
        channel = request.path_params["channel"]
        root = _studio_project(request, write=request.method == "POST")
        if request.method == "POST":
            body = await _json_body(request)
            if not isinstance(body, dict) or set(body) - {"table", "folders", "roles"}:
                raise ValueError("Choose a combined table or its folders.")
            if "folders" in body:
                await run_in_threadpool(phenotyping_inputs.save_folders, root, body["folders"])
            if "table" in body:
                await run_in_threadpool(phenotyping_inputs.use_table, channel, body["table"])
            if "roles" in body:
                project = registry.session_project(channel) or {}
                await run_in_threadpool(phenotyping_inputs.save_roles, root, project.get("unified"), body["roles"])
        project = registry.session_project(channel) or {}
        return JSONResponse(await run_in_threadpool(phenotyping_inputs.plan, root, project.get("mappings", {}), project.get("unified")))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_resolve_setup(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import resolve_setup
    from spacemap_gui.studio.phenotyping_inputs import combined_table
    try:
        body = await _json_body(request) if request.method == "POST" else {}
        action = body.get("action", "get")
        root = _studio_project(request, write=action == "save")
        table = combined_table(request.path_params["channel"])
        if action == "get":
            result = await run_in_threadpool(resolve_setup.get, root, table)
        elif action == "save":
            result = await run_in_threadpool(resolve_setup.save, root, table, body)
        elif action == "validate":
            from spacemap import dataio
            markers = resolve_setup.validate_options(body.get("values", {}), dataio.columns(table))
            result = {"panel": resolve_setup.rules_payload(body.get("rules", ""), markers)}
        elif action in {"calculate", "background"}:
            result = await run_in_threadpool(resolve_setup.calculate, root, table, body.get("values", {}),
                                             background_folder=body.get("folder", "") if action == "background" else None)
        elif action == "draft":
            result = await run_in_threadpool(resolve_setup.draft_reference, root, table, body)
        else:
            raise ValueError("Choose a listed RESOLVE action.")
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_phenotyping_link(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import cell_links
    try:
        _studio_project(request)
        body = await _json_body(request)
        tokens = body.get("layers") or []
        if not isinstance(tokens, list) or len(tokens) > 32 or any(not isinstance(t, str) for t in tokens):
            raise ValueError("Choose the open cell layers.")
        return JSONResponse(await run_in_threadpool(cell_links.context, request.path_params["channel"], tokens))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_reference_features(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import reference_features, auto_reference
    try:
        _studio_project(request, write=request.method == "POST")
        channel = request.path_params["channel"]
        if request.method == "POST":
            body = await _json_body(request)
            result = await run_in_threadpool(reference_features.save, channel, body.get("features"), body.get("allow_missing", True))
            previous = await run_in_threadpool(auto_reference.status, channel)
            if previous.get("enabled"):
                result["auto_reference"] = await run_in_threadpool(auto_reference.configure, channel,
                    previous["column"], True, previous["checked_only"])
        else:
            result = await run_in_threadpool(reference_features.options, channel)
        return JSONResponse(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)


async def studio_auto_reference(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import auto_reference
    try:
        _studio_project(request, write=request.method == "POST")
        channel = request.path_params["channel"]
        if request.method == "POST":
            body = await _json_body(request)
            result = await run_in_threadpool(auto_reference.configure, channel, body.get("column"), body.get("enabled"), body.get("checked_only", False))
        else:
            result = await run_in_threadpool(auto_reference.status, channel)
        return JSONResponse(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)


async def studio_manual_reference(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import manual_reference, saved
    try:
        channel = request.path_params["channel"]
        body = await _json_body(request) if request.method == "POST" else {}
        action = body.get("action", "options")
        _studio_project(request, write=action in {"create", "assign", "undo", "write"})
        if action == "options":
            result = await run_in_threadpool(manual_reference.options, channel, body.get("column") or request.query_params.get("column", ""))
        elif action == "create":
            result = await run_in_threadpool(manual_reference.create, channel, body.get("column"))
        elif action == "assign":
            result = await run_in_threadpool(manual_reference.assign, channel, body)
        elif action == "undo":
            result = await run_in_threadpool(manual_reference.undo, channel, body.get("request_id"))
        elif action == "evaluate":
            path, _, _ = manual_reference.context(channel)
            result = await run_in_threadpool(manual_reference.evaluate, path, body.get("reference"), body.get("method"))
        elif action == "write":
            result = await run_in_threadpool(saved.act, registry.session_project(channel), "write", {})
        else:
            raise ValueError("Choose a manual reference action.")
        return JSONResponse(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _data_source(body, channel=None):
    """The table a data view reads (a cell layer's table, or a file brought in) and its image rows' default."""
    source = body.get("source") or {}
    if source.get("combined") and channel:
        from spacemap_gui.studio import phenotyping_inputs
        project = registry.session_project(channel) or {}
        path = phenotyping_inputs.combined_table(channel)
        roles = phenotyping_inputs.column_roles(project["root"], path)
        return path, {"column": roles["roi"], "value": None,
                      "label_column": phenotyping_inputs.segmentation_column(project["root"], path), "cell_id_column": roles["cell_id"]}
    if source.get("cells"):
        from spacemap_gui.studio.cell_links import table_layer
        layer = registry.resolve_object(str(source["cells"]), "cells") or {}
        layer = table_layer(layer, str(source.get("table", "")))
        path = (layer.get("tables") or {}).get(str(source.get("table", "")))
        if not path:
            raise ValueError("Choose a table of these cells.")
        return path, {"column": layer.get("roi_column", "batch_id"), "value": layer.get("roi"),
                      "label_column": layer.get("label_column", "label"), "cell_id_column": layer.get("cell_id_column", "cell_id")}
    path = (registry.resolve_object(str(source.get("file", "")), "datafile") or {}).get("path")
    if not path or not Path(path).is_file():
        raise ValueError("Open the table again (drop it, or type its path).")
    return path, {"column": "", "value": None}


async def studio_data(request: Request) -> Response:
    """The data viewer: info, query (a page), find (a cell's row), labels (the matching cells), edit, save, export."""
    import json as _json

    from starlette.concurrency import run_in_threadpool
    from starlette.responses import FileResponse
    from starlette.background import BackgroundTask

    from spacemap_gui.studio import data_view, data_editor

    try:
        action = request.path_params["action"]
        _studio_project(request, write=action in ("save",))
        body = await _json_body(request) if request.method == "POST" else _json.loads(request.query_params.get("q", "{}"))
        path, default = _data_source(body, request.path_params["channel"])
        if action == "status":
            return JSONResponse(data_editor.status(path))
        if action == "discard":
            return JSONResponse(await run_in_threadpool(data_view.history, path, action))
        for role in ("column", "label_column", "cell_id_column"):
            if default.get(role):
                default[role] = await run_in_threadpool(data_view.current_name, path, default[role])
        rows = body.get("rows") or {}
        column = rows.get("column", default["column"]) if rows.get("this_image", True) else ""
        value = rows.get("value", default["value"]) if column else None
        filters, sort = body.get("filters") or [], body.get("sort") or None
        if action == "info":  # the image column's values, to pick this image's rows in a merged table
            found = await run_in_threadpool(data_view.info, path, rows.get("column") or default["column"])
            return JSONResponse({**found, "default": str(default["value"]) if default["value"] is not None else "",
                                 "roi_column": default["column"], "label_column": default.get("label_column", "label"), "cell_id_column": default.get("cell_id_column", "cell_id")})
        if action == "summary":
            return JSONResponse(await run_in_threadpool(data_view.summary, path, column, value, filters, str(body.get("column", "")),
                                                        int(body.get("limit", 30)), int(body.get("bins", 24))))
        if action == "query":
            found = await run_in_threadpool(data_view.query, path, column, value, filters, sort,
                                           body.get("offset", 0), body.get("limit", data_view.PAGE), body.get("columns"))
            from spacemap.viewer import cells
            ids = [row[0] for row in found["rows"]]
            fields = {"label": default.get("label_column", "label"), "roi": default["column"],
                      "cell_id": default.get("cell_id_column", "cell_id"), "group": body.get("selection_column", "")}
            identity = {}
            for key, name in fields.items():
                if name in found["columns"]:
                    identity[key] = [data_view._cell(v) for v in await run_in_threadpool(data_view.values_at_rows, path, ids, name)]
            found["identity"] = {str(row): {key: values[i] for key, values in identity.items()} for i, row in enumerate(ids)}
            return JSONResponse(found)
        if action == "find":
            return JSONResponse({"offset": await run_in_threadpool(data_view.find, path, column, value, filters, sort,
                                int(body.get("label", -1)), default.get("label_column", "label"), default["column"], body.get("cell_roi", default["value"]))})
        if action == "labels":
            return JSONResponse({"labels": await run_in_threadpool(data_view.labels_of, path, column, value, filters, default.get("label_column", "label"))})
        if action == "groups":
            from spacemap.viewer import cells
            if not column or value is None:
                raise ValueError("Choose an ROI before linking selected cells.")
            _, positions = await run_in_threadpool(cells.assignment_rows, path, str(value), body.get("labels") or [],
                                                  roi_column=column, label_column=default.get("label_column", "label"))
            values = await run_in_threadpool(cells.values_at_rows, path, positions, str(body.get("column", "")))
            return JSONResponse({"groups": sorted({str(v) for v in values if v is not None})})
        if action in ("edit", "mutate"):
            operation = body.get("operation") or {"kind": "cells", "changes": body.get("changes") or []}
            # These identifiers connect table rows to image masks. Independent
            # copies remain editable; a linked table keeps its identity intact.
            protected = {default.get(k) for k in ("column", "label_column", "cell_id_column")} if body.get("source", {}).get("cells") or body.get("source", {}).get("combined") else set()
            touched = set(operation.get("columns") or []) if operation.get("kind") == "delete_columns" else {c.get("col") for c in operation.get("changes") or []} if operation.get("kind") == "cells" else {operation.get("column")}
            if touched & protected and operation.get("kind") != "delete_rows":
                raise ValueError("This column links cells to images. Save as an independent copy to change its values or header.")
            return JSONResponse(await run_in_threadpool(data_view.mutate, path, operation, column, value, filters, body.get("selected"), body.get("revision")))
        if action in ("undo", "redo"):
            return JSONResponse(await run_in_threadpool(data_view.history, path, action))
        if action == "save":
            linked = body.get("source", {}).get("cells") or body.get("source", {}).get("combined")
            folder = Path(_studio_project(request)) / "outputs/studio/data_edits" if linked else None
            saved = await run_in_threadpool(data_view.save, path, body.get("mode", "copy"), folder, body.get("name"), body.get("revision"))
            token = registry.register_object("datafile", path=saved)
            return JSONResponse({"saved": saved, "file_token": token, "name": Path(saved).name})
        if action == "export":
            name = Path(path).stem + "_" + (str(value) + "_" if value else "") + "rows.csv"
            exported = await run_in_threadpool(data_editor.export_file, path, column, value, filters, sort, body.get("columns"))
            return FileResponse(exported, media_type="text/csv", filename=name,
                                background=BackgroundTask(Path(exported).unlink, missing_ok=True))
        raise ValueError("Choose a data action.")
    except data_view.Preparing as exc:
        return JSONResponse({"preparing": True, "message": str(exc)})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_spatial(request: Request) -> Response:
    """Spatial analysis and Visualization on the combined table: quick results at once, long ones as jobs."""
    from starlette.concurrency import run_in_threadpool

    import json

    from spacemap_gui.studio import data_view, jobs, spatial, tls

    try:
        action = request.path_params["action"]
        root = Path(_studio_project(request, write=action in ("start", "tls_save") or (action == "tls_settings" and request.method == "POST")))
        project = registry.session_project(request.path_params["channel"]) or {}
        table = project.get("unified")
        if not table or not Path(table).is_file():
            raise ValueError("No combined table yet: make it in Batch correction.")
        body = await _json_body(request) if request.method == "POST" else {}
        images = [str(i) for i in body.get("images") or []] or None
        text = lambda key: str(body.get(key) or "")
        def tls_image_entries():
            channel = request.path_params["channel"]
            spec = (registry.session_view(channel, mark_seen=False) or {}).get("spec") or {}
            entries = []
            for layer in spec.get("layers", []):
                if layer.get("type") != "image":
                    continue
                path = registry.resolve(layer.get("token", ""))
                sample = next((c.get("roi") for c in spec.get("layers", []) if c.get("type") == "cells"
                               and c.get("image_token") == layer.get("token") and c.get("roi")), None)
                if path:
                    entries.append({"path": str(path), "sample": sample, "name": Path(path).name})
            return entries + (registry.session_images(channel) or [])
        if action == "tls_mpp":
            from spacemap_gui.studio import phenotyping_overview
            overview = await run_in_threadpool(phenotyping_overview.overview, table)
            return JSONResponse({"images": await run_in_threadpool(tls.calibration, root,
                [r["roi"] for r in overview.get("rois", [])], tls_image_entries())})
        if action == "tls_settings":
            from spacemap_gui.studio.settings import read_config, revision
            chosen = (await run_in_threadpool(tls.save, root, body.get("changes"), body.get("revision"))
                      if request.method == "POST" else tls.values(read_config(root)))
            return JSONResponse({"values": chosen, "revision": revision(chosen)})
        if action == "tls_labels":
            return JSONResponse({"labels": await run_in_threadpool(tls.labels, table, text("column"))})
        if action == "tls_result":
            return JSONResponse({"result": await run_in_threadpool(tls.latest, root, table)})
        if action in ("tls_edit", "tls_trim", "tls_save"):
            from spacemap_gui.studio import tls_review
            if action == "tls_save":
                settings = await run_in_threadpool(tls_review.prepare, root, table, body)
                return JSONResponse({"job": await run_in_threadpool(jobs.start, str(root), "tls_candidates", settings)})
            draft = await run_in_threadpool(tls_review.context, root, table, text("job_id"), text("image"), text("region"))
            if action == "tls_trim":
                return JSONResponse({"features": await run_in_threadpool(tls_review.trim, body.get("features"), body.get("amount"), draft["pixel_um"])})
            return JSONResponse(draft)
        if action in ("tls_shapes", "tls_download"):
            job_id = text("job_id") or request.query_params.get("job_id", "")
            job = await run_in_threadpool(jobs.status, root, job_id)
            if job["kind"] != "tls_candidates" or job["status"] != "done" or job.get("result", {}).get("table") != str(table):
                raise ValueError("Choose a completed TLS candidate run for this table.")
            folder = jobs.directory(root, job_id)
            if action == "tls_download":
                from starlette.responses import FileResponse
                extension = request.query_params.get("format", "geojson")
                if extension not in ("geojson", "csv"):
                    raise ValueError("Choose CSV or GeoJSON.")
                return FileResponse(folder / ("tls_candidates." + extension), filename="tls_candidates." + extension)
            found = json.loads((folder / "tls_candidates.geojson").read_text())
            found["features"] = [f for f in found["features"] if f["properties"]["image"] == text("image")
                                 and (not text("key") or f["properties"]["key"] == text("key"))]
            return JSONResponse(found)
        if action == "info":
            def info():
                names = spatial.columns(table)
                groups = [c for c in names if c.lower().startswith(("phenotype", "cluster", "neighbo", "leiden", "consensus")) or c.endswith("_cn")]
                return {"table": Path(table).name, "columns": names, "groups": groups, "distances": [c for c in names if c.endswith("_distance")],
                        "markers": spatial.markers_of(table), "umaps": spatial.saved_umaps(root)}
            return JSONResponse(await run_in_threadpool(info))
        if action == "composition":
            return JSONResponse(await run_in_threadpool(spatial.composition, table, text("column"), images))
        if action == "profile":
            return JSONResponse(await run_in_threadpool(spatial.profile, table, text("column"), body.get("markers") or [], images))
        if action == "distance":
            return JSONResponse(await run_in_threadpool(spatial.distance_profile, table, text("column"), text("distance"), images,
                                                        float(body.get("step") or 25), float(body.get("limit") or 300), body.get("inside", True) is not False))
        if action == "agreement":
            return JSONResponse(await run_in_threadpool(spatial.agreement, table, text("reference"), text("method"), images))
        if action == "umap_points":
            return JSONResponse(await run_in_threadpool(spatial.umap_points, root, text("name"), table, text("column")))
        if action == "triad_shapes":
            folder = Path(text("folder")).resolve()
            points = folder / f"triad_instances_{folder.name}.csv"
            if root.resolve() not in folder.parents or not points.is_file():
                raise ValueError("Choose a triad run of this project.")
            return JSONResponse(await run_in_threadpool(spatial.triad_shapes, points, text("image")))
        if action == "outline":
            folder = Path(text("folder")).resolve()
            path = folder / (text("image") + "_tissue_outline.geojson")
            if root.resolve() not in folder.parents or "/" in text("image") or not path.is_file():
                raise ValueError("No tissue outline for this image in that run.")
            return JSONResponse(json.loads(path.read_text()))
        if action == "start":
            kind = text("kind")
            if kind == "tls_candidates":
                from spacemap_gui.studio.settings import read_config, revision
                chosen = tls.values(read_config(root))
                if body.get("settings_revision") != revision(chosen):
                    raise ValueError("TLS settings changed. Save the settings before running.")
                if not isinstance(body.get("overwrite", False), bool):
                    raise ValueError("Choose whether to overwrite the result columns.")
                settings = await run_in_threadpool(tls.prepare, root, table, chosen, images, body.get("overwrite", False), tls_image_entries())
                return JSONResponse({"job": await run_in_threadpool(jobs.start, str(root), kind, settings)})
            if kind not in ("spatial_density", "spatial_triads", "spatial_screening", "spatial_umap"):
                raise ValueError("Choose an analysis.")
            settings = {key: value for key, value in body.items() if key not in ("kind", "root", "table", "images")}
            job = await run_in_threadpool(jobs.start, str(root), kind, {**settings, "root": str(root), "table": str(table), "images": images or []})
            return JSONResponse({"job": job})
        raise ValueError("Choose an action.")
    except data_view.Preparing as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


SERVER_PORT = None


async def studio_servers(request: Request) -> Response:
    """This user's SPACEMAP servers (GET), and stopping one left behind on this machine (POST {host, port})."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import servers

    try:
        if request.method == "POST":
            body = await _json_body(request)
            done = await run_in_threadpool(servers.stop, str(body.get("host", "")), int(body.get("port", -1)))
            return JSONResponse({"done": done, "servers": servers.listing()})
        return JSONResponse({"servers": await run_in_threadpool(servers.listing)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_freshness(request: Request) -> Response:
    """Results older than what they were made from: tagged tables before their segmentation, the combined table
    before an image's segmentation or tagging."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import freshness
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        project = registry.session_project(request.path_params["channel"]) or {}
        config = read_config(root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        return JSONResponse(await run_in_threadpool(freshness.check, root, config, mappings, project.get("unified")))
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_cell_data(request: Request) -> Response:
    """Cell data tables on the open image. GET: the annotations rows can be kept inside and the data layers it has.
    POST inspect {file_token}; build {file_token, choice}; remove {id}. GET ?export=<id> downloads a layer's circles or
    Voronoi cells as GeoJSON."""
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import FileResponse

    from spacemap_gui.studio import annotation, cell_data, cell_layers
    from spacemap_gui.studio.settings import read_config

    from . import ui

    try:
        channel = request.path_params["channel"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(channel) or {}
        body = await _json_body(request) if request.method == "POST" else {}
        entry, image_id = _open_entry(channel, body.get("image_token") or request.query_params.get("image_token", ""))
        config = read_config(root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        if request.method == "GET":
            found = cell_data.records(root, entry["path"])
            if request.query_params.get("export"):
                item = next((i for i in found if i["id"] == request.query_params["export"] and i.get("geojson")), None)
                if item is None:
                    raise ValueError("That layer has no circles or Voronoi cells to export.")
                name = Path(item["name"]).stem + "_" + item["show"] + ".geojson"
                return FileResponse(item["geojson"], media_type="application/geo+json", filename=name)
            output = annotation.folders(root, mappings)[1]
            areas = await run_in_threadpool(cell_data.area_choices, root, config, entry["path"], output)
            own = await run_in_threadpool(cell_layers.own, root, config, mappings, entry["path"], entry["sample"])
            return JSONResponse({"areas": areas, "segmentation": bool(own),
                                 "layers": [{"id": i["id"], "name": i["name"], "show": i["show"], "text": cell_data.describe(i)} for i in found]})
        action = body.get("action")
        if action == "remove":
            await run_in_threadpool(cell_data.remove, root, entry["path"], str(body.get("id", "")))
            return JSONResponse({"removed": True})
        if action == "rebuild":  # the same table with a new size, reach or Within
            old = next((i for i in cell_data.records(root, entry["path"]) if i["id"] == str(body.get("id", ""))), None)
            if old is None:
                raise ValueError("That data layer is not on this image any more.")
            changes = {k: v for k, v in (body.get("changes") or {}).items() if k in ("size_um", "reach_um", "within")}
            own = await run_in_threadpool(cell_layers.own, root, config, mappings, entry["path"], entry["sample"])
            record = await run_in_threadpool(cell_data.build, root, config, mappings, entry["path"], old["source"],
                                             {**old["choice"], **changes}, ui._cache_dir(root), own)
            if record["id"] != old["id"]:
                await run_in_threadpool(cell_data.remove, root, entry["path"], old["id"])
            layers = await run_in_threadpool(_layers_of, channel, entry)
            layer = next((item for item in layers if item.get("id") == "data-" + record["id"]), None)
            return JSONResponse({"record": {**record, "text": cell_data.describe(record)}, "layer": layer, "replaces": "data-" + old["id"]})
        source = (registry.resolve_object(str(body.get("file_token", "")), "datafile") or {}).get("path")
        if not source or not Path(source).is_file():
            raise ValueError("Bring the table in again (drop it, or type its path).")
        if action == "inspect":
            return JSONResponse(await run_in_threadpool(cell_data.inspect, source, entry["sample"]))
        if action == "build":
            own = await run_in_threadpool(cell_layers.own, root, config, mappings, entry["path"], entry["sample"])
            record = await run_in_threadpool(cell_data.build, root, config, mappings, entry["path"], source,
                                             dict(body.get("choice") or {}), ui._cache_dir(root), own)
            layers = await run_in_threadpool(_layers_of, channel, entry)
            wanted = "cells" if record["show"] == "cells" else "data-" + record["id"]
            layer = next((item for item in layers if item.get("id") == wanted), None)
            return JSONResponse({"record": {**record, "text": cell_data.describe(record)}, "layer": layer})
        raise ValueError("Choose inspect, build or remove.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_cell_layers(request: Request) -> Response:
    """The open image's cell segmentations: GET lists the project's files for it (to import) and the imported ones;
    POST {action: add, image_token, name | path} imports one as a layer, {action: remove, image_token, path_token} takes
    it off again (the file stays)."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer.catalog import sample_of
    from spacemap_gui.studio import cell_layers
    from spacemap_gui.studio.settings import read_config

    from . import ui

    try:
        channel = request.path_params["channel"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(channel) or {}
        body = await _json_body(request) if request.method == "POST" else {}
        token = body.get("image_token") or request.query_params.get("image_token", "")
        entry, image_id = _open_entry(channel, token)
        config = read_config(root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        found = await run_in_threadpool(cell_layers.candidates, root, config, mappings, entry["path"], entry["sample"])
        if request.method == "GET":
            return JSONResponse({"files": [{"name": item["name"]} for item in found],
                                 "imported": [Path(p).name for p in cell_layers.imported(root, entry["path"])]})
        action = body.get("action")
        if action == "add":
            if body.get("name"):
                chosen = next((item["path"] for item in found if item["name"] == body["name"]), None)
                if chosen is None:
                    raise ValueError("That file is not in the list any more. Refresh it.")
            elif body.get("file_token"):  # brought in by drop or path
                chosen = (registry.resolve_object(str(body["file_token"]), "datafile") or {}).get("path", "")
            else:
                chosen = str(body.get("path") or "").strip()  # a full path typed by the user
            table = (registry.resolve_object(str(body["table_token"]), "datafile") or {}).get("path") if body.get("table_token") else None
            path = await run_in_threadpool(cell_layers.add, root, entry["path"], chosen, table)
            layers = await run_in_threadpool(_layers_of, channel, entry)
            layer = next((item for item in layers if item.get("imported_cells") and
                          Path((registry.resolve_object(item["token"], "cells") or {}).get("segmentation", "")).resolve() == Path(path)), None)
            if layer is None:
                raise ValueError("These outlines are already shown as the image's cells.")
            return JSONResponse({"layer": {**layer, "visible": True}})
        if action == "remove":
            source = registry.resolve_object(str(body.get("layer_token", "")), "cells") or {}
            if not source.get("segmentation"):
                raise ValueError("Choose an imported cell layer.")
            await run_in_threadpool(cell_layers.remove, root, entry["path"], source["segmentation"])
            return JSONResponse({"removed": True})
        raise ValueError("Choose add or remove.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_model_library(request: Request) -> Response:
    """The model library (where it is, what it holds) and this project's trained models; POST changes the library's
    folder, shares a project model into it, or imports a library model into the project."""
    from starlette.concurrency import run_in_threadpool

    import os

    from spacemap_gui.studio import model_library

    try:
        body = await _json_body(request) if request.method == "POST" else {}
        action = body.get("action")
        root = Path(_studio_project(request, write=action in ("share", "import")))
        project = registry.session_project(request.path_params["channel"]) or {}
        done = None
        if action == "location":
            await run_in_threadpool(model_library.set_location, body.get("path"))
        elif action == "share":
            done = await run_in_threadpool(model_library.share, root, body.get("kind"), body.get("id"), project.get("name") or root.name)
        elif action == "import":
            done = await run_in_threadpool(model_library.import_model, root, body.get("kind"), body.get("entry"))
        elif action is not None:
            raise ValueError("Choose Share, Import or the library folder.")
        return JSONResponse({"location": str(model_library.location()), "kinds": model_library.KINDS,
                             "fixed": bool(os.environ.get(model_library.ENVIRONMENT)),
                             "project": await run_in_threadpool(model_library.project_models, root),
                             "library": await run_in_threadpool(model_library.entries), "done": done})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_projects(request: Request) -> Response:
    """The projects classic SPACEMAP knows and the open project's notes; POST opens one (by name or folder), makes a
    new one, takes one off the list, or saves the open project's notes."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import projects

    channel = request.path_params["channel"]
    try:
        if registry.session_view(channel, mark_seen=False) is None:
            raise ValueError("Open Studio from SPACEMAP first.")
        body = await _json_body(request) if request.method == "POST" else {}
        action, done = body.get("action"), None
        if action == "open":
            done = await run_in_threadpool(lambda: projects.open_project(channel, name=body.get("name"), folder=body.get("folder")))
        elif action == "create":
            done = await run_in_threadpool(projects.create, channel, body.get("name"), body.get("parent"))
        elif action == "forget":
            await run_in_threadpool(projects.forget, body.get("name"))
        elif action == "notes":
            done = await run_in_threadpool(projects.save_notes, Path(_studio_project(request, write=True)), body.get("changes"))
        elif action is not None:
            raise ValueError("Choose Open, New project or Remove from list.")
        root = (registry.session_project(channel) or {}).get("root")
        parent = Path(root).parent if root else Path(__file__).resolve().parents[3] / "project"
        return JSONResponse({"projects": await run_in_threadpool(projects.known, root), "open": root,
                             "notes": await run_in_threadpool(projects.notes, Path(root)) if root else None,
                             "new_parent": str(parent), "done": done})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_annotation_images(request: Request) -> Response:
    """The Annotation step's folders and images with the stages done for each; POST {changes} saves folders and label."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(request.path_params["channel"]) or {}
        if request.method == "POST":
            body = await _json_body(request)
            changes = body.get("changes")
            if isinstance(changes, dict) and set(changes) == {"subfolder"} and isinstance(changes["subfolder"], dict):
                value = changes["subfolder"]
                config = await run_in_threadpool(read_config, root)
                mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
                moved = await run_in_threadpool(annotation.save_subfolder, root, mappings, value.get("on"), value.get("path"),
                                                bool(value.get("move")))
                plan = await run_in_threadpool(annotation.images_plan, root, mappings, read_config(root))
                return JSONResponse({**plan, "moved": len(moved["moved"])})
            await run_in_threadpool(annotation.save_step, root, changes)
        config = await run_in_threadpool(read_config, root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        return JSONResponse(await run_in_threadpool(annotation.images_plan, root, mappings, config))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_roi_images(request: Request) -> Response:
    """The ROI step's folders and images (ROI file, ROIs, extracted, core settings); ?image= adds that image's
    markers. POST {changes} saves the step's settings; POST {image, markers} saves one image's markers."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import rois
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(request.path_params["channel"]) or {}

        def places():
            config = read_config(root)
            mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
            return config, mappings, rois.all_folders(root, mappings, config)

        def image_path(name):
            _, _, folders = places()
            path = folders["images"] / str(name or "")
            if not name or "/" in str(name) or not path.is_file():
                raise ValueError(f"{name} is not in the images folder.")
            return path

        if request.method == "POST":
            body = await _json_body(request)
            if "markers" in body:
                chosen = body.get("markers") or {}
                await run_in_threadpool(rois.save_markers, root, image_path(body.get("image")), chosen.get("keep"),
                                        chosen.get("rename") or {}, chosen.get("colors") or {})
            else:
                await run_in_threadpool(rois.save_step, root, body.get("changes"))
        config, mappings, _ = await run_in_threadpool(places)
        found = await run_in_threadpool(rois.images_plan, root, mappings, config)
        image = request.query_params.get("image")
        if image:
            from spacemap.viewer.display import settings_path
            path = await run_in_threadpool(image_path, image)
            display = settings_path(root, path)
            found["markers"] = await run_in_threadpool(rois.markers, root, path, str(display) if display.is_file() else None)
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_roi_markers(request: Request) -> Response:
    """The open image's markers as Tab 3 extracts them (kept, new names); POST saves Tab 3's choice for the image."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import rois

    try:
        channel = request.path_params["channel"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        view = registry.session_view(channel, mark_seen=False) or {}
        image = next(
            (layer for layer in (view.get("spec") or {}).get("layers", []) if layer.get("type") == "image"), None
        )
        path = registry.resolve(image["token"]) if image else None
        if not path:
            raise ValueError("Open an image first.")
        if request.method == "POST":
            body = await _json_body(request)
            found = await run_in_threadpool(rois.save_markers, root, path, body.get("keep"), body.get("rename") or {}, body.get("colors") or {})
        else:
            from spacemap.viewer.display import settings_path
            display = settings_path(root, path)
            found = await run_in_threadpool(rois.markers, root, path, str(display) if display.is_file() else None)
        return JSONResponse({"image": Path(path).name, "markers": found})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_registration_pairs(request: Request) -> Response:
    """Tab 1's pairs for the project (naming rule, per-image choices, folders); POST saves changes to them."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            found = await run_in_threadpool(registration.save_pairing, root, body.get("changes"), body.get("revision"))
        else:
            config = await run_in_threadpool(read_config, root)
            found = await run_in_threadpool(registration.plan, root, config.get("directory_mappings") or {}, config)
        found = {**found, "revision": registration.pairing_revision(found)}
        found["rows"] = await run_in_threadpool(_pair_images, request.path_params["channel"], found)
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _pair_images(channel, found):
    """Each pair's CODEX, H&E and merged image as the window can open them (group and name, no paths)."""
    from spacemap_gui.viewer.ui import window_images

    images = window_images(channel)
    entries = registry.session_images(channel) or []
    by_path = {}
    for image, entry in zip(images, entries):
        try:
            by_path[str(Path(entry["path"]).resolve())] = {"group": image["group"], "name": image["name"]}
        except (KeyError, OSError):
            pass
    folders = found["folders"]
    rows = []
    for row in found["rows"]:
        where = {"codex": (folders["codex"], row["codex"]), "he": (folders["he"], row["he"]), "merged": (folders["registered"], row["output"])}
        look = lambda folder, name: by_path.get(str((Path(folder) / name).resolve())) if name else None  # noqa: E731
        rows.append({**row, "images": {kind: look(folder, name) for kind, (folder, name) in where.items()},
                     "extra_images": [look(folder, item["file"]) for folder, item in zip(found.get("extra_folders", []), row.get("extra", []))]})
    return rows


async def studio_registration_markers(request: Request) -> Response:
    """One pair's layers (?sample=): each CODEX image's markers and the H&E's; POST {"sample", "edits"} saves them."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            return JSONResponse(await run_in_threadpool(registration.save_pair_markers, root, body.get("sample"), body.get("edits")))
        config = await run_in_threadpool(read_config, root)
        return JSONResponse(await run_in_threadpool(registration.pair_markers, root, config.get("directory_mappings") or {},
                                                    config, request.query_params.get("sample", "")))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_registration_rotation(request: Request) -> Response:
    """POST {"sample", "rotation"}: the H&E's turn set on the image, saved with the pair."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration

    try:
        root = Path(_studio_project(request, write=True))
        body = await _json_body(request)
        return JSONResponse(await run_in_threadpool(registration.save_pair_rotation, root, body.get("sample"), body.get("rotation")))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_registration_quick_align(request: Request) -> Response:
    """POST {"sample"}: where the pair's H&E sits on its CODEX, from small copies (seconds), to place the overlay."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        body = await _json_body(request)
        config = await run_in_threadpool(read_config, root)
        return JSONResponse(await run_in_threadpool(registration.quick_align, root, config.get("directory_mappings") or {}, config, body.get("sample", "")))
    except (ValueError, OSError, TypeError, KeyError, StopIteration) as exc:
        return JSONResponse({"error": str(exc) or "No such pair."}, status_code=400)


async def studio_registration_figure(request: Request) -> Response:
    """The matched-points figure of a pair's registration (?sample=), as a picture for the figure panel."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration, tab8_results
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        sample = request.query_params.get("sample", "")
        config = await run_in_threadpool(read_config, root)
        path = await run_in_threadpool(registration.feature_matches, root, config.get("directory_mappings") or {}, config, sample)
        if path is None:
            return JSONResponse({"sample": sample, "image": None})
        return JSONResponse({"sample": sample, "name": path.name, "image": await run_in_threadpool(tab8_results._picture, path)})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_registration_quality(request: Request) -> Response:
    """How well the open registered image's H&E sits on its CODEX (DAPI against the H&E nuclei)."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import registration

    try:
        view = registry.session_view(request.path_params["channel"], mark_seen=False) or {}
        image = next(
            (layer for layer in (view.get("spec") or {}).get("layers", []) if layer.get("type") == "image"), None
        )
        path = registry.resolve(image["token"]) if image else None
        if not path:
            raise ValueError("Open a registered image first.")
        return JSONResponse({"image": Path(path).name, **(await run_in_threadpool(registration.quality, path))})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_tab8_results(request: Request) -> Response:
    """What classic Tab 8 saved: triad tables by type, and a feature screening run's strongest findings."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import tab8_results

    try:
        root = Path(_studio_project(request))
        run = request.query_params.get("run") or None
        return JSONResponse(
            {
                "triads": await run_in_threadpool(tab8_results.triads, root),
                "screening": await run_in_threadpool(tab8_results.screening, root, run),
            }
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_batch_checks(request: Request) -> Response:
    """How well Tab 6's batch correction worked, from the combined table Studio shows."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import batches
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        project = registry.session_project(request.path_params["channel"]) or {}
        if not project.get("unified"):
            raise ValueError("No combined table yet: run Tab 6 first.")
        return JSONResponse(await run_in_threadpool(batches.check, project["unified"], read_config(root)))
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_project_status(request: Request) -> Response:
    """The Project step: samples by step, folders, the project file and the assistant."""
    from spacemap_gui.studio import assistant
    from spacemap_gui.studio.steps import project_status

    try:
        return JSONResponse({**project_status(request.path_params["channel"]), "assistant": assistant.status()})
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_background_images(request: Request) -> Response:
    """The Background step's folders and images (annotation, classes, blank cycles, state); POST saves its settings."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import background
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            await run_in_threadpool(background.save_step, root, body.get("changes"))
        config = await run_in_threadpool(read_config, root)
        project = registry.session_project(request.path_params["channel"]) or {}
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        found = await run_in_threadpool(background.images_plan, root, mappings, config)
        from spacemap.background_detection import THRESHOLD_METHODS

        found["methods"] = list(THRESHOLD_METHODS)
        image = request.query_params.get("image")
        if image:
            found["markers"] = await run_in_threadpool(background.image_markers, root, mappings, config, image)
            row = next((r for r in found["rows"] if r["image"] == image), None)
            if row and row.get("annotation"):
                image_path = str(Path(found["folders"]["images"]) / image)
                annotation_path = str(Path(found["folders"]["annotations"]) / row["annotation"])
                found["measurement_annotation"] = {
                    "type": "annotations", "name": row["annotation"],
                    "token": registry.register_object("annotations", image=image_path, path=annotation_path),
                    "image_token": registry.register(image_path), "classes": row.get("classes", []),
                }
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_background_levels(request: Request) -> Response:
    """The open image's levels saved in Tab 2's threshold table, if any."""
    from spacemap_gui.studio import background, commands
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        channel = request.path_params["channel"]
        spec = registry.session_view(channel, mark_seen=False)["spec"]
        _, image, _ = commands._view(channel, spec, {}, [])
        project = registry.session_project(channel) or {}
        config = read_config(root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        _, tables, removed = background.folders(root, mappings, config)
        name = Path(registry.resolve(image["token"])).name
        table = background.table_path(tables, name)
        return JSONResponse(
            {
                "image": name,
                "table": table.name,
                "levels": background.read_levels(table) if table.is_file() else None,
                "removed": background.output_path(removed, name).is_file(),
            }
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_segmentation_settings(request: Request) -> Response:
    """Tab 5's settings for Studio: read, or save as classic Tab 5 does."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import segmentation
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(segmentation.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = segmentation.values(root, read_config(root))
        return JSONResponse({"values": saved, "revision": revision(saved), "models": list(segmentation.MODELS)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_segmentation_images(request: Request) -> Response:
    """The Segmentation step's folders, images (where each is segmented, done or not, how many cells) and models;
    POST {changes} saves folders, the model (by id) or an image's area."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import segmentation
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(request.path_params["channel"]) or {}
        if request.method == "POST":
            body = await _json_body(request)
            await run_in_threadpool(segmentation.save_plan, root, body.get("changes"))
        config = await run_in_threadpool(read_config, root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        return JSONResponse(await run_in_threadpool(segmentation.plan, root, config, mappings))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_segmentation_regions(request: Request) -> Response:
    """Where to segment: each ROI's annotation file (by name, or chosen by hand), the classes, and the folders to choose from."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import segmentation, steps, tagging
    from spacemap_gui.studio.settings import read_config

    try:
        root = Path(_studio_project(request))
        config = await run_in_threadpool(read_config, root)
        project = registry.session_project(request.path_params["channel"]) or {}
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        _, keys, fallback = steps.LAYOUT["tab5"][0][0]
        names, _ = steps._files(steps._folder(root, mappings, keys, fallback))
        names = [name for name in names if Path(name).suffix.lower() in {".tif", ".tiff", ".qptiff", ".svs"}]
        settings = segmentation.values(root, config)
        found = await run_in_threadpool(segmentation.regions, settings, names)
        found["folder"] = tagging._shown(root, settings["tab4_annot_geojson_dir"])
        found["folders"] = await run_in_threadpool(tagging.folders_with, root, ".geojson")
        found["tagging_rules"] = len(settings["tab4_spatial_rules"]) if settings["tab4_spatial_enabled"] else 0
        # The annotations saved in the Annotation step, by name, and how many images have each.
        saved = await run_in_threadpool(segmentation.saved_annotations, root, config, mappings, names)
        counts = {}
        for row in found["rows"]:
            row["saved"] = sorted(saved.get(row["image"], {}))
            for name in row["saved"]:
                counts[name] = counts.get(name, 0) + 1
        found["layers"] = [{"name": name, "images": count} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))]
        found["layer"] = settings["tab4_mask_layer"]
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_training(request: Request) -> Response:
    """Cellpose training: list the tiles and session; start a session, keep, delete, move or validate tiles."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import commands, segmentation, training
    from spacemap_gui.studio.settings import read_config

    try:
        action = request.path_params.get("action", "")
        root = Path(_studio_project(request, write=request.method == "POST" and action != "validate"))
        if request.method == "GET":
            config = await run_in_threadpool(read_config, root)
            project = registry.session_project(request.path_params['channel']) or {}
            mappings = {**project.get('mappings', {}), **config.get('directory_mappings', {})}
            found = {**await run_in_threadpool(training.examples, root),
                     **await run_in_threadpool(training.starting_models, root, config, mappings),
                     "limit": training.REGION_LIMIT, "open_image": None}
            try:  # the open image's file name, to draw its tiles
                channel = request.path_params["channel"]
                _, image, _ = commands._view(channel, registry.session_view(channel, mark_seen=False)["spec"], {}, [])
                found["open_image"] = Path(registry.resolve(image["token"])).name
            except (ValueError, KeyError, TypeError):
                pass
            if request.query_params.get("outlines") and found["open_image"]:
                # The corrected cells of the kept tiles on the open image, to show inside the tiles.
                found["outlines"] = await run_in_threadpool(training.tile_outlines, root, found["open_image"])
            return JSONResponse(found)
        body = await _json_body(request)
        if action == "start":
            return JSONResponse(await run_in_threadpool(training.start, root, body.get("name"), body.get("base")))
        if action == "delete":
            return JSONResponse(await run_in_threadpool(training.delete_example, root, str(body.get("name", ""))))
        if action == "role":
            return JSONResponse(await run_in_threadpool(training.set_role, root, body.get("names"), body.get("role")))
        if action == "validate":
            return JSONResponse(await run_in_threadpool(training.validate, root))
        if action == "scores":
            # Every version on the same validation tiles: a job, as it runs each model.
            from spacemap_gui.studio import jobs

            found = training.examples(root)
            if not found["session"]:
                raise ValueError("Start a training session first.")
            if not found["counts"]["validate"]:
                raise ValueError("Keep at least one corrected validation tile first.")
            gpu = bool(segmentation.values(root, read_config(root))["tab4_gpu"])
            return JSONResponse({"job": jobs.start(root, "score_cellpose", {"project_root": str(root), "gpu": gpu})})
        if action != "examples":
            raise ValueError("Unknown training action.")
        channel = request.path_params["channel"]
        spec = registry.session_view(channel, mark_seen=False)["spec"]
        _, image, info = commands._view(channel, spec, {}, [])
        settings = {**segmentation.values(root, read_config(root)), **segmentation.check(body.get("settings") or {})}
        bounds = training.check_region(body.get("bounds"), info.width, info.height)
        path = registry.resolve(image["token"])
        return JSONResponse(
            await run_in_threadpool(
                lambda: training.save_example(
                    root,
                    path,
                    list(info.channel_names),
                    bounds,
                    body.get("polygons"),
                    settings,
                    role=body.get("role", "train"),
                    predicted=body.get("predicted"),
                    model=str(body.get("model") or ""),
                )
            )
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_samples(request: Request) -> Response:
    """Samples: the ROIs with their slides and the study's own columns; edit them or link a table."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import samples
    from spacemap_gui.studio.settings import read_config

    try:
        action = request.path_params.get("action", "")
        root = Path(_studio_project(request, write=request.method == "POST" and action not in {"suggest", "preview"}))
        project = registry.session_project(request.path_params["channel"]) or {}
        mappings = {**project.get("mappings", {}), **read_config(root).get("directory_mappings", {})}
        if request.method == "GET":
            return JSONResponse(await run_in_threadpool(samples.table, root, mappings))
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        if action == "edit":
            work = lambda: samples.edit(root, mappings, body.get("roi"), body.get("column"), body.get("value"))  # noqa: E731
        elif action == "add":
            work = lambda: samples.add(root, mappings, body.get("column"))  # noqa: E731
        elif action == "rename":
            work = lambda: samples.rename(root, mappings, body.get("column"), body.get("name"))  # noqa: E731
        elif action == "drop":
            work = lambda: samples.drop(root, mappings, body.get("column"))  # noqa: E731
        elif action == "suggest":
            work = lambda: samples.suggest_link(root, mappings, body.get("text"))  # noqa: E731
        elif action in {"preview", "link"}:
            work = lambda: samples.link(  # noqa: E731
                root,
                mappings,
                body.get("text"),
                body.get("link_column"),
                body.get("link_to"),
                body.get("columns"),
                apply=action == "link",
            )
        else:
            raise ValueError("Unknown samples action.")
        if action not in {"suggest", "preview"}:
            from spacemap_gui.studio.settings import project_lock
            operation = work
            def work():
                with project_lock(root):
                    return operation()
        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_correction(request: Request) -> Response:
    """Batch correction: what a run would use (GET); save the choices and run it as a job; use its table."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import correction, jobs
    from spacemap_gui.studio.settings import read_config

    try:
        action = request.path_params.get("action", "")
        channel = request.path_params["channel"]
        root = Path(_studio_project(request, write=request.method == "POST"))
        project = registry.session_project(channel) or {}
        config = read_config(root)
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        if request.method == "GET":
            found = await run_in_threadpool(correction.plan, root, mappings, config, project.get("unified"))
            return JSONResponse({**found, "table": Path(project["unified"]).name if project.get("unified") else ""})
        body = await _json_body(request)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON.")
        if action == "folders":
            config = await run_in_threadpool(correction.save_folders, root, body.get("folders"))
            mappings = {**mappings, **config.get("directory_mappings", {})}
            return JSONResponse(await run_in_threadpool(correction.plan, root, mappings, config, project.get("unified")))
        if action == "save":
            saved = await run_in_threadpool(correction.save, root, body.get("changes") or {}, body.get("markers"))
            return JSONResponse({"settings": saved})
        if action == "run":
            plan = await run_in_threadpool(correction.plan, root, mappings, config)
            markers = body.get("markers")
            if not isinstance(markers, list) or not markers or set(markers) - set(plan["common"]):
                raise ValueError("Choose markers that every cell table has.")
            saved = await run_in_threadpool(correction.save, root, body.get("changes") or {}, markers)
            rois = body.get("rois")
            if rois is not None and not (isinstance(rois, list) and all(isinstance(r, str) for r in rois)):
                raise ValueError("Tick the tables to correct.")
            settings = await run_in_threadpool(correction.job_settings, root, mappings, saved, markers, rois)
            return JSONResponse({"job": jobs.start(root, "correct_batches", settings)})
        if action == "use":
            job = jobs.status(root, str(body.get("job_id", "")))
            if job.get("kind") != "correct_batches" or job.get("status") != "done":
                raise ValueError("Choose a finished batch correction.")
            output = Path(job["result"]["output"])
            # The path comes from this project's completed job, including a user-chosen external output folder.
            if not output.is_file():
                raise ValueError("That saved correction table is no longer available.")
            from spacemap_gui.studio.phenotyping_inputs import use_table
            await run_in_threadpool(use_table, channel, str(output))
            return JSONResponse({"table": output.name})
        raise ValueError("Unknown batch correction action.")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_colours(request: Request) -> Response:
    """Phenotype colours shared with Tab 9: read them, or save some and recolour the cells."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import colours
    from spacemap_gui.studio.settings import read_config
    from spacemap import phenotype_presets

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "GET":
            return JSONResponse({"colours": colours.saved(read_config(root)),
                                 "phenotypes":[item['name'] for item in phenotype_presets.read()['entries']]})
        body = await _json_body(request)
        if isinstance(body, dict) and 'names' in body:
            names = body['names']
            if not isinstance(names, list) or len(names) > 500 or any(not isinstance(name, str) or not 0 < len(name.strip()) <= 128 for name in names):
                raise ValueError('Choose up to 500 names, each of 1–128 characters.')
            mapping = await run_in_threadpool(phenotype_presets.ensure, names)
            return JSONResponse({'mapping':mapping, 'colours':colours.saved(read_config(root)),
                                 'phenotypes':[item['name'] for item in phenotype_presets.read()['entries']]})
        found = await run_in_threadpool(colours.save, root, body.get("colours") if isinstance(body, dict) else None)
        registry.use_colours(request.path_params["channel"], found, str(root))
        _LUTS.clear()  # the cached colour tables were made with the old colours
        _RESOLVE_LUTS.clear()
        return JSONResponse({"colours": found})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_phenotype_presets(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap import phenotype_presets
    from spacemap_gui.studio import colours
    try:
        root = _studio_project(request, write=request.method == "POST")
        if request.method == "GET":
            return JSONResponse(await run_in_threadpool(phenotype_presets.read))
        body = await _json_body(request)
        result = await run_in_threadpool(phenotype_presets.save, body.get("entries"), body.get("revision"))
        from spacemap_gui.studio.settings import read_config
        registry.use_colours(request.path_params["channel"], colours.saved(read_config(root)), str(root))
        _LUTS.clear()
        _RESOLVE_LUTS.clear()
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({"error":str(exc)}, status_code=400)


async def studio_segment_preview(request: Request) -> Response:
    """Segment part of the open image with Tab 5's Cellpose settings; nothing is saved."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import commands, segmentation
    from spacemap_gui.studio.settings import read_config

    try:
        channel = request.path_params["channel"]
        body = await _json_body(request)
        root = Path(_studio_project(request))
        spec = registry.session_view(channel, mark_seen=False)["spec"]
        _, image, info = commands._view(channel, spec, {}, [])
        settings = {**segmentation.values(root, read_config(root)), **segmentation.check(body.get("settings") or {})}
        box = body.get("bounds")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            raise ValueError("Choose the area to test.")
        x0, y0 = max(0.0, box[0]), max(0.0, box[1])
        x1, y1 = min(float(info.width), box[0] + box[2]), min(float(info.height), box[1] + box[3])
        # A large view is tested in its centre, at most 1024 x 1024 pixels.
        side = segmentation.PREVIEW_LIMIT
        if x1 - x0 > side:
            x0 = (x0 + x1 - side) / 2
            x1 = x0 + side
        if y1 - y0 > side:
            y0 = (y0 + y1 - side) / 2
            y1 = y0 + side
        path = registry.resolve(image["token"])
        result = await run_in_threadpool(
            segmentation.preview, path, list(info.channel_names), [x0, y0, x1 - x0, y1 - y0], settings
        )
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_tissue_settings(request: Request) -> Response:
    """Tab 4's tissue settings for Studio: read, or save into Tab 4's saved config."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(annotation.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = annotation.settings(read_config(root))
        from spacemap.tissue_annotation_utils import TISSUE_MASK_MODES

        return JSONResponse({"values": saved, "revision": revision(saved), "modes": list(TISSUE_MASK_MODES)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_tissue_preview(request: Request) -> Response:
    """The same detection for preview clients and the auto-saving This image action."""
    import time as clock

    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, commands
    from spacemap_gui.studio.settings import read_config

    try:
        channel = request.path_params["channel"]
        body = await _json_body(request)
        saving = body.get('save') is True
        root = Path(_studio_project(request, write=saving))
        spec = registry.session_view(channel, mark_seen=False)["spec"]
        _, image, _ = commands._view(channel, spec, {}, [])
        config = read_config(root)
        values = {**annotation.settings(config), **annotation.check(body.get("settings") or {})}
        started = clock.perf_counter()
        from spacemap_gui.viewer import ui as viewer_ui

        image_path = registry.resolve(image["token"])
        project = registry.session_project(channel) or {}
        mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
        _, output = annotation.folders(root, mappings)
        target = annotation.valid_tissue(output, annotation.sample_id(image_path))
        from spacemap_gui.studio import annotation_edits, annotation_inputs
        if body.get('image_token') and registry.resolve(body['image_token']) != image_path:
            raise ValueError('The image changed. Detect tissue again on the open image.')
        before, version = await run_in_threadpool(annotation_edits.read, target, kind='tissue', label=annotation.tissue_label(config))
        if saving and body.get('revision') != version:
            raise ValueError('Tissue changed. Wait for its edits to save, then detect again.')
        found = await run_in_threadpool(annotation.detect, image_path, config, values,
            **({'output':output,'expected':version} if saving else {}))
        if saving:
            await run_in_threadpool(annotation_inputs.save_selection, root, config, image_path, output, 'tissue', {'mode':'tissue','class_name':''})
        layer = viewer_ui.shapes_layer(target, image_path, name=annotation.tissue_label(config), layer_id="tissue-regions",
                                      regions=True, annotation_kind="tissue", tissue_label=annotation.tissue_label(config),
                                      editable=not (registry.session_project(channel) or {}).get("read_only", False),
                                      reference=target)
        return JSONResponse(
            {
                "outlines": annotation.rings(found["features"], include_holes=True),
                # The preview can be pruned and saved without running detection
                # again. Keep the full outlines so that save is lossless.
                "features": found["features"],
                "layer": layer,
                "pieces": len(found["features"]),
                "stats": found["stats"],
                "seconds": round(clock.perf_counter() - started, 2),
                **({'revision':found['revision'], 'before':annotation_edits.display(before,version)['features'], 'saved':True} if saving else {}),
            }
        )
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _open_annotation_image(request):
    """(root, config, image path, Tab 4 boundary file) for the open image."""
    from spacemap_gui.studio import annotation, commands
    from spacemap_gui.studio.settings import read_config

    root = Path(_studio_project(request))
    channel = request.path_params["channel"]
    spec = registry.session_view(channel, mark_seen=False)["spec"]
    _, image, _ = commands._view(channel, spec, {}, [])
    config = read_config(root)
    project = registry.session_project(channel) or {}
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    _, output = annotation.folders(root, mappings)
    path = registry.resolve(image["token"])
    return root, config, path, annotation.boundary(output, annotation.sample_id(Path(path).name))


async def studio_artifact_settings(request: Request) -> Response:
    """The artifact sliders as Tab 4 has them saved; saving writes Tab 4's own settings."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import artifacts
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(artifacts.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = artifacts.values(read_config(root))
        ranges = {
            part: {"min": spec[3][0], "max": spec[3][1], "step": spec[4]} for part, spec in artifacts.PARTS.items()
        }
        return JSONResponse(
            {"values": saved, "revision": revision(saved), "ranges": ranges, "colours": artifacts.COLOURS}
        )
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_artifact_preview(request: Request) -> Response:
    """One artifact's live mask for a slider value on the open image; nothing is saved."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import artifacts

    try:
        body = await _json_body(request)
        _, config, path, boundary = _open_annotation_image(request)
        part, value = body.get("part"), body.get("value")
        artifacts.check({part: {"value": value}} if part in artifacts.PARTS else {"?": {}})
        return JSONResponse(await run_in_threadpool(artifacts.preview, path, config, part, float(value), boundary))
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_artifact_example(request: Request) -> Response:
    """A slider value that catches the drawn example; nothing is saved."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import artifacts

    def rings(value):
        if value is None:
            return None
        if not (isinstance(value, list) and 0 < len(value) <= 20):
            raise ValueError("Draw one to twenty example areas.")
        for ring in value:
            if not (
                isinstance(ring, list)
                and 3 <= len(ring) <= 4096
                and all(
                    isinstance(point, list) and len(point) == 2 and all(isinstance(n, (int, float)) for n in point)
                    for point in ring
                )
            ):
                raise ValueError("An example area needs at least three corners.")
        return value

    try:
        body = await _json_body(request)
        _, config, path, boundary = _open_annotation_image(request)
        return JSONResponse(
            await run_in_threadpool(
                artifacts.suggest,
                path,
                config,
                body.get("part"),
                rings(body.get("examples")),
                rings(body.get("clean")),
                boundary,
            )
        )
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_pixel_classes(request: Request) -> Response:
    """Tab 4's pixel classes for the open image at a low pyramid level; nothing is saved.

    With ``domains`` ({class: name}) the map is coloured by domain, as the names stand
    in the panel, saved or not. The names saved for this image come back as ``saved``.
    """
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, pixelclasses
    from spacemap_gui.studio.settings import revision

    try:
        body = await _json_body(request)
        root, config, path, boundary = _open_annotation_image(request)
        sample = annotation.sample_id(Path(path).name)
        from spacemap_gui.studio import annotation_inputs
        valid = annotation_inputs.scope_path(root, path, boundary.parent.parent, "classes")
        hidden, excluded = body.get("hidden") or [], body.get("excluded") or []
        for numbers in (hidden, excluded):
            if not (isinstance(numbers, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in numbers)):
                raise ValueError("Hide or remove classes by number.")
        domains = body.get("domains")
        found = await run_in_threadpool(
            pixelclasses.classify, path, config, body.get("n_clusters"), valid, hidden, domains, excluded
        )
        saved = pixelclasses.names(config, sample, found["n_clusters"])
        return JSONResponse({**found, "sample": sample, "saved": {**saved, "revision": revision(saved)}})
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_class_names(request: Request) -> Response:
    """Save the open image's class names into Tab 4's domain settings."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, pixelclasses

    try:
        body = await _json_body(request)
        _studio_project(request, write=True)
        root, _, path, _ = _open_annotation_image(request)
        saved = await run_in_threadpool(
            pixelclasses.save_names,
            root,
            annotation.sample_id(Path(path).name),
            body.get("n_clusters"),
            body.get("domains"),
            body.get("excluded"),
            body.get("revision"),
        )
        return JSONResponse(saved)
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_pieces_settings(request: Request) -> Response:
    """Tab 4's tissue piece settings (step 2b): read, or save into Tab 4's saved config."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import pieces
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(pieces.save, root, body.get("changes"), body.get("revision"))
        else:
            saved = pieces.settings(read_config(root))
        return JSONResponse({"values": saved, "revision": revision(saved)})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_pieces_preview(request: Request) -> Response:
    """Tab 4's tissue pieces for the open image with the sent settings, drawn but not saved."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, pieces

    try:
        body = await _json_body(request)
        root, config, path, boundary = _open_annotation_image(request)
        changes = pieces.check(body.get("changes") or {})
        values = {**pieces.settings(config), **changes}
        # Bind a preview to the Tissue layer the user is actually reviewing.
        # Saved layers are read at full precision, not from simplified display outlines.
        if body.get("image_token") and registry.resolve(body["image_token"]) != path:
            raise ValueError("The image changed. Preview necks again on the open image.")
        output = boundary.parent.parent
        tissue_path = annotation.valid_tissue(output, annotation.sample_id(Path(path).name))
        expected = None
        tissue = None
        if body.get("tissue_token"):
            tissue = registry.resolve_object(body["tissue_token"], "shapes")
            if not tissue or tissue.get("annotation_kind") not in ("tissue", "regions") or Path(tissue["image"]).resolve() != Path(path).resolve():
                raise ValueError("Open this image's Tissue layer, then preview necks again.")
            if not isinstance(body.get("revision"), str):
                raise ValueError("Wait for Tissue to load, then preview necks again.")
            tissue_path, expected = Path(tissue["path"]), body["revision"]
        from spacemap_gui.studio import annotation_inputs
        chosen = annotation_inputs.selection(root,path,"tissue")
        from spacemap.viewer.sources import image_info
        info = await run_in_threadpool(image_info, path)
        values["microns_per_pixel"] = info.pixel_size_um or values["microns_per_pixel"]
        found = await run_in_threadpool(pieces.preview, output, annotation.sample_id(Path(path).name), values,
                                       features=body.get("features"), label=annotation.tissue_label(config),
                                       tissue_path=tissue_path, expected=expected, class_name=chosen.get("class_name", ""))
        found['layer'] = annotation_inputs.working_layer(root,config,path,output,readonly=bool(
            (registry.session_project(request.path_params['channel']) or {}).get('read_only')))
        if Path(found['layer']['saved_file']).resolve() != tissue_path.resolve():
            # A loaded/editable Tissue layer may come from another saved file.
            from spacemap_gui.viewer.ui import shapes_layer
            found['layer'] = shapes_layer(tissue_path,path,name=annotation.tissue_label(config),layer_id='tissue-regions',
                regions=True,annotation_kind=(tissue or {}).get('annotation_kind','tissue'),
                tissue_label=annotation.tissue_label(config),reference=tissue_path,
                editable=not (registry.session_project(request.path_params['channel']) or {}).get('read_only',False))
        return JSONResponse(found)
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_zone_rules(request: Request) -> Response:
    """Tab 4's pool and zone rules (step 4): read, or save into Tab 4's saved config."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import zones
    from spacemap_gui.studio.settings import read_config, revision

    try:
        root = Path(_studio_project(request, write=request.method == "POST"))
        if request.method == "POST":
            body = await _json_body(request)
            saved = await run_in_threadpool(zones.save, root, body.get("rules"), body.get("revision"))
        else:
            saved = zones.values(read_config(root))
        config = read_config(root)
        return JSONResponse(
            {"values": saved, "revision": revision(saved), "microns_per_pixel": zones.microns_per_pixel(config)}
        )
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_zones_preview(request: Request) -> Response:
    """Tab 4's pools and zones for the open image with the sent rules, drawn but not saved."""
    from starlette.concurrency import run_in_threadpool

    from spacemap_gui.studio import annotation, zones

    try:
        body = await _json_body(request)
        _, config, path, boundary = _open_annotation_image(request)
        output = boundary.parent.parent
        from spacemap.tissue_annotation_utils import PyramidImageReader

        shape = PyramidImageReader(path).full_shape
        return JSONResponse(
            await run_in_threadpool(
                zones.preview, output, annotation.sample_id(Path(path).name), config, body.get("rules") or {}, shape
            )
        )
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_selection_options(request: Request) -> Response:
    from spacemap import dataio
    from spacemap_gui.studio import commands, selection
    from spacemap.viewer import reference

    try:
        root = _studio_project(request)
        channel = request.path_params["channel"]
        spec = registry.session_view(channel, mark_seen=False)["spec"]
        spec, image, info = commands._view(channel, spec, {}, [])
        layer, registered, table = commands._cell_table(spec, image)
        available = dataio.columns(table)
        return JSONResponse(
            {
                "markers": [marker for marker in info.channel_names if marker in available],
                "columns": list(dict.fromkeys([column for column in available if column.startswith("phenotype_")] + reference.columns(table))),
                "phenotypes": reference.phenotype_names(table) if "phenotype_ml_consensus" in available else [],
                "rules": selection.library(root),
                "measurements": [column for column in available
                                 if column not in info.channel_names
                                 and column not in {"cell_id", "label", "batch_id", *commands._keys(registered).values()}
                                 and not column.startswith("phenotype_")],
            }
        )
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_settings(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import settings

    try:
        root = _studio_project(request, write=request.method == "POST")
        if request.method == "GET":
            return JSONResponse(await run_in_threadpool(settings.get, root))
        body = await _json_body(request)
        return JSONResponse(await run_in_threadpool(settings.save, root, body.get("changes"), body.get("revision")))
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def studio_heatmap_summary(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool
    from spacemap_gui.studio import cluster_profiles
    try:
        root = _studio_project(request, write=True)
        body = await _json_body(request)
        if body.get("aggregation") not in ("median", "mean"):
            raise ValueError("Choose Median or Mean for the heatmap.")
        aggregation = await run_in_threadpool(cluster_profiles.preference, root, body.get("method"), body.get("aggregation"))
        return JSONResponse({"aggregation": aggregation})
    except (ValueError, OSError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def studio_jobs(request: Request) -> Response:
    from spacemap_gui.studio import jobs

    try:
        root = _studio_project(request, write=request.method == "POST")
        job_id = request.path_params.get("job_id")
        if request.method == "POST":
            return JSONResponse(jobs.cancel(root, job_id))
        if job_id:
            return JSONResponse({"job": jobs.status(root, job_id), "log": jobs.log(root, job_id)})
        from spacemap_gui.studio import auto_reference
        try:
            auto = auto_reference.tick(request.path_params["channel"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            auto = {"state":"error", "message":str(exc)}
        return JSONResponse({"jobs": jobs.recent(root), "auto_reference":auto})
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def session_get(request: Request) -> Response:
    view = registry.session_view(request.path_params["session"])
    if view is None:
        return JSONResponse(
            {"error": "This viewer window is no longer linked. Open it again from SPACEMAP."}, status_code=404
        )
    known = request.query_params.get("v")
    if known and known == str(view["version"]):
        return JSONResponse({"version": view["version"]})
    return JSONResponse(view)


def session_images(request: Request) -> Response:
    from . import ui

    session = request.path_params["session"]
    if registry.session_view(session) is None:
        return JSONResponse(
            {"error": "This viewer window is no longer linked. Open it again from SPACEMAP."}, status_code=404
        )
    try:
        return JSONResponse({"images": ui.window_images(session)})
    except Exception as exc:
        return JSONResponse({"error": f"Could not list the project's images: {exc}"}, status_code=500)


async def session_add_image(request: Request) -> Response:
    """Add an image from outside the project to the window's list (checked: it must open as an image)."""
    from starlette.concurrency import run_in_threadpool

    from . import ui

    session = request.path_params["session"]
    if registry.session_view(session) is None:
        return JSONResponse(
            {"error": "This viewer window is no longer linked. Open it again from SPACEMAP."}, status_code=404
        )
    try:
        body = await _json_body(request)
        result = await run_in_threadpool(ui.window_add_image, session, str(body.get("path") or ""))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(result)


async def session_open(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    from . import ui

    try:
        body = await _json_body(request)
        image_id = int(body.get("id"))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc) or "Choose an image."}, status_code=400)
    try:
        result = await run_in_threadpool(ui.window_open_image, request.path_params["session"], image_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"Could not open the image: {exc}"}, status_code=500)
    return JSONResponse(result)


# ---------------------------------------------------------------- live RESOLVE


def _resolve_spec(request: Request):
    spec = registry.resolve_object(request.path_params["token"], "resolve")
    if spec is None:
        return None, JSONResponse({"error": "Unknown RESOLVE session. Reopen the viewer."}, status_code=404)
    return spec, None


def _resolve_session(spec: dict):
    from spacemap.viewer import live_resolve

    return live_resolve.open_session(spec)


def _resolve_state(spec: dict, session) -> dict:
    from spacemap.viewer import live_resolve

    return {
        **live_resolve.summary(session),
        "roi": session.roi,
        "markers": [
            {"marker": m, "threshold": session.thresholds[m], "source": session.source[m]} for m in session.markers
        ],
        "phenotypes": list(session.engine.names),
        "saved_as": Path(spec.get("thresholds") or "").name,
    }


async def resolve_open(request: Request) -> Response:
    """Start (or rejoin) live RESOLVE for this ROI: reads its cells once, then keeps them in memory."""
    from starlette.concurrency import run_in_threadpool

    spec, error = _resolve_spec(request)
    if error:
        return error
    try:
        session = await run_in_threadpool(_resolve_session, spec)
    except (OSError, ValueError, KeyError) as exc:
        return JSONResponse({"error": f"Live RESOLVE could not start: {exc}"}, status_code=400)
    return JSONResponse(_resolve_state(spec, session))


async def resolve_thresholds(request: Request) -> Response:
    """New thresholds (raw marker values): phenotypes are assigned again, as Tab 7 would."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import live_resolve

    spec, error = _resolve_spec(request)
    if error:
        return error
    try:
        body = await _json_body(request)
        changes = {str(k): float(v) for k, v in (body.get("thresholds") or {}).items()}
        session = await run_in_threadpool(_resolve_session, spec)
        return JSONResponse(await run_in_threadpool(live_resolve.assign, session, changes))
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


_RESOLVE_LUTS: dict = {}


def _resolve_lut(spec: dict, session, hidden: frozenset, near=False):
    from spacemap.viewer import live_resolve

    from spacemap.phenotype_presets import signature
    key = (id(session), session.version, hidden, near, signature())
    cached = _RESOLVE_LUTS.get(key)
    if cached is None:
        from spacemap.phenotype_presets import ensure
        palette = {**(spec.get("colors") or {}), **ensure(session.result.counts)}
        cached = live_resolve.phenotype_lut(session, colors=palette, hidden=set(hidden), near=near)
        if len(_RESOLVE_LUTS) > 16:
            _RESOLVE_LUTS.clear()
        _RESOLVE_LUTS[key] = cached
    return cached


def resolve_tile(request: Request) -> Response:
    """Cells coloured by the live phenotypes (the URL carries the version, so tiles can be cached)."""
    import numpy as np

    from spacemap.viewer import cells

    spec, error = _resolve_spec(request)
    if error:
        return error
    layer = registry.resolve_object(spec["cells"], "cells")
    if layer is None:
        return JSONResponse({"error": "Unknown cell layer. Reopen the viewer."}, status_code=404)
    if not cells.pyramid_ready(layer["pyramid"]):
        empty = _png(np.zeros((1, 1, 4), dtype=np.uint8))
        empty.headers["Cache-Control"] = "no-store"
        return empty
    session = _resolve_session(spec)
    params = request.query_params
    hidden = frozenset(h for h in params.get("hide", "").split("\x1f") if h)
    lut, _ = _resolve_lut(spec, session, hidden, near=params.get("near") == "1")
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    outline = params.get("ol", "ffff00")
    labels = cells.label_tile(layer["pyramid"], image_info(layer["image"]), z, x, y)
    return _png(
        cells.render_cell_tile(
            labels,
            lut,
            mode=params.get("mode", "fill"),
            outline_rgba=tuple(int(outline[i : i + 2], 16) for i in (0, 2, 4)) + (255,),
            z=z,
        )
    )


def resolve_legend(request: Request) -> Response:
    spec, error = _resolve_spec(request)
    if error:
        return error
    session = _resolve_session(spec)
    hidden = frozenset(h for h in request.query_params.get("hide", "").split("\x1f") if h)
    _, legend = _resolve_lut(spec, session, hidden)
    return JSONResponse({"version": session.version, "items": legend})


def resolve_histogram(request: Request) -> Response:
    from spacemap.viewer import live_resolve

    spec, error = _resolve_spec(request)
    if error:
        return error
    marker = request.query_params.get("marker", "")
    session = _resolve_session(spec)
    if marker not in session.markers:
        return JSONResponse({"error": f"{marker} is not one of this panel's markers."}, status_code=404)
    if "transforms" not in request.query_params:
        return JSONResponse(live_resolve.histogram(session, marker))
    from spacemap.viewer import transforms

    try:
        names = [n for n in request.query_params["transforms"].split(",") if n]
        transform = live_resolve.histogram_transform(
            session, marker, names, float(request.query_params.get("cofactor", 5))
        )
        if "auto" in request.query_params:
            return JSONResponse(
                transforms.automatic_threshold(
                    session.raw[:, session.markers.index(marker)], transform, request.query_params["auto"], params=dict(request.query_params)
                )
            )
        if "value" in request.query_params:
            raw = float(transform.inverse(float(request.query_params["value"])))
            return JSONResponse({"raw": raw})
        return JSONResponse(
            live_resolve.transformed_histogram(session, marker, names, float(request.query_params.get("cofactor", 5)))
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def resolve_why(request: Request) -> Response:
    """Why the cell under a point (full-resolution pixels) got its phenotype."""
    from spacemap.viewer import live_resolve

    spec, error = _resolve_spec(request)
    if error:
        return error
    layer = registry.resolve_object(spec["cells"], "cells")
    try:
        x, y = float(request.query_params["x"]), float(request.query_params["y"])
    except (KeyError, ValueError):
        return JSONResponse({"error": "Give x and y."}, status_code=400)
    label = _label_at(layer, x, y) if layer else 0
    if not label:
        return JSONResponse({"label": 0})
    try:
        return JSONResponse(live_resolve.explain(_resolve_session(spec), label))
    except ValueError as exc:
        return JSONResponse({"label": label, "error": str(exc)})


async def resolve_save(request: Request) -> Response:
    """Write this ROI's thresholds into RESOLVE's threshold table (the file Tab 7 reads)."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import live_resolve

    spec, error = _resolve_spec(request)
    if error:
        return error
    if not spec.get("thresholds"):
        return JSONResponse({"error": "This project has no RESOLVE threshold table location."}, status_code=400)
    session = await run_in_threadpool(_resolve_session, spec)
    path = await run_in_threadpool(live_resolve.save, session)
    return JSONResponse({"saved": path.name, "roi": session.roi})


async def resolve_panel(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    from spacemap import dataio
    from spacemap.viewer import panels

    spec, error = _resolve_spec(request)
    if error:
        return error
    try:
        available = await run_in_threadpool(dataio.columns, spec["table"])
        if request.method == "GET":
            return JSONResponse({**await run_in_threadpool(panels.read, spec["panel"]), "markers": list(available)})
        body = await _json_body(request)

        def save_and_refresh():
            # The engine loads scientific libraries on first use. A simple
            # panel read must not start it; a save does so off the event loop.
            from spacemap.viewer import live_resolve

            saved = panels.save(spec["panel"], body.get("panel"), body.get("revision"), available)
            live_resolve.invalidate_panel(spec["panel"])
            session = _resolve_session(spec)
            return {**saved, "live": _resolve_state(spec, session)}

        return JSONResponse(await run_in_threadpool(save_and_refresh))
    except (OSError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def display_get(request: Request) -> Response:
    from spacemap.viewer import display
    from .settings_io import run as settings_io

    layer = registry.resolve_object(request.path_params["token"], "display")
    if layer is None:
        return JSONResponse({"error": "Unknown image settings. Reopen the viewer."}, status_code=404)
    return JSONResponse(await settings_io(display.load, layer["path"]))


async def display_save(request: Request) -> Response:
    from .settings_io import run as settings_io

    from spacemap.viewer import display

    layer = registry.resolve_object(request.path_params["token"], "display")
    if layer is None:
        return JSONResponse({"error": "Unknown image settings. Reopen the viewer."}, status_code=404)
    try:
        body = await _json_body(request)
        saved = await settings_io(display.save, layer["path"], body.get("channels"), body.get("auto"), body.get("stamp"), layer.get('image'))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not save the settings: {exc}"}, status_code=500)
    return JSONResponse({"saved_at": saved["saved_at"]})


async def display_sets(request: Request) -> Response:
    """The project's marker sets: GET lists them; POST saves {name, markers} or deletes {delete: name}."""
    from starlette.concurrency import run_in_threadpool

    from spacemap.viewer import display

    layer = registry.resolve_object(request.path_params["token"], "display")
    if layer is None:
        return JSONResponse({"error": "Unknown image settings. Reopen the viewer."}, status_code=404)
    path = display.sets_path(layer["path"])
    if request.method == "GET":
        return JSONResponse({"sets": display.load_sets(path)})
    try:
        body = await _json_body(request)
        if body.get("delete"):
            sets = await run_in_threadpool(display.delete_set, path, body["delete"])
        else:
            sets = await run_in_threadpool(display.save_set, path, body.get("name"), body.get("markers"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"sets": sets})


routes = [
    Route("/studio/{channel}/classification", classification_api.classification, methods=["GET", "POST"]),
    Route("/studio/{channel}/heatmap_summary", studio_heatmap_summary, methods=["POST"]),
    Route("/studio/{channel}/annotation_export", annotation_export_api.annotation_export, methods=["GET", "POST"]),
    Route("/studio/{channel}/annotation_names", annotation_export_api.annotation_names_route, methods=["GET", "POST"]),
    Route("/studio/{channel}/annotation_palette", annotation_export_api.annotation_palette, methods=["GET", "POST"]),
    Route("/studio/{channel}/annotation_imports", annotation_export_api.annotation_imports, methods=["GET", "POST"]),
    Route("/studio/{channel}/cell_layers", studio_cell_layers, methods=["GET", "POST"]),
    Route("/studio/{channel}/cell_data", studio_cell_data, methods=["GET", "POST"]),
    Route("/studio/{channel}/freshness", studio_freshness),
    Route("/studio/{channel}/servers", studio_servers, methods=["GET", "POST"]),
    Route("/studio/{channel}/data/{action}", studio_data, methods=["GET", "POST"]),
    Route("/studio/{channel}/phenotyping_overview", studio_phenotyping_overview),
    Route("/studio/{channel}/phenotyping_inputs", studio_phenotyping_inputs, methods=["GET", "POST"]),
    Route("/studio/{channel}/resolve_setup", studio_resolve_setup, methods=["GET", "POST"]),
    Route("/studio/{channel}/phenotyping_link", studio_phenotyping_link, methods=["POST"]),
    Route("/studio/{channel}/bring/{action}", studio_bring, methods=["POST"]),
    Route("/studio/{channel}/annotation_difference", annotation_difference_api.annotation_difference, methods=["GET", "POST"]),
    Route("/studio/{channel}/artifact_review/{part}/{action}", artifact_api.review, methods=["GET", "POST"]),
    Route("/studio/{channel}/annotation_sources", artifact_api.annotation_sources, methods=["GET", "POST"]),
    Route("/studio/{channel}/remove_small_pieces", artifact_api.remove_small_pieces, methods=["POST"]),
    Route("/studio/{channel}/spatial/{action}", studio_spatial, methods=["GET", "POST"]),
    Route("/studio/{channel}/tissue_versions", artifact_api.tissue_versions, methods=["GET", "POST"]),
    Route("/health", health),
    Route("/image/{token}/info", info),
    Route("/image/{token}/limits", limits),
    Route("/image/{token}/background", background),
    Route("/image/{token}/display_range", auto_display_range),
    Route("/image/{token}/histogram", histogram),
    Route("/image/{token}/threshold_benchmark", threshold_benchmark),
    Route("/image/{token}/tile/{z:int}/{x:int}/{y:int}.jpg", tile),
    Route("/image/{token}/tile/{z:int}/{x:int}/{y:int}.png", tile),
    Route("/cells/{token}/tile/{z:int}/{x:int}/{y:int}.png", cell_tile),
    Route("/cells/{token}/legend", cell_legend),
    Route("/cells/{token}/ready", cell_ready),
    Route("/cells/{token}/columns", cell_columns),
    Route("/cells/{token}/at", cell_at),
    Route("/cells/{token}/outline", cell_outline),
    Route("/cells/{token}/profile", cell_profile, methods=["POST"]),
    Route("/cells/{token}/suggestions", cell_suggestions, methods=["POST"]),
    Route("/cells/{token}/reference_models", cell_reference_models),
    Route("/cells/{token}/filter", cell_filter_count),
    Route("/cells/{token}/reference-options", cell_reference_options),
    Route("/cells/{token}/review", cell_review, methods=["GET", "POST"]),
    Route("/cells/{token}/assign", cell_assign, methods=["POST"]),
    Route("/annotations/{token}/tile/{z:int}/{x:int}/{y:int}.png", annotation_tile),
    Route("/annotations/{token}/legend", annotation_legend),
    Route("/annotations/{token}/at", annotation_at),
    Route("/annotations/{token}/profile", annotation_profile, methods=["POST"]),
    Route("/shapes/edit", shapes_edit, methods=["POST"]),
    Route("/shapes/{token}", shapes_get),
    Route("/shapes/{token}/save", shapes_save, methods=["POST"]),
    Route("/shapes/{token}/brush", shapes_brush, methods=["POST"]),
    Route("/shapes/{token}/edit", shapes_polygon_edit, methods=["POST"]),
    Route("/shapes/{token}/history", shapes_history, methods=["GET", "POST"]),
    Route("/shapes/{token}/sources", shapes_sources),
    Route("/shapes/{token}/profile", shapes_profile, methods=["POST"]),
    Route("/shapes/{token}/import", shapes_import, methods=["POST"]),
    Route("/shapes/{token}/download", shapes_download),
    Route("/cells/{token}/density/{z:int}/{x:int}/{y:int}.png", density_tile),
    Route("/cells/{token}/density/legend", density_legend),
    Route("/cells/{token}/density/groups", density_groups),
    Route("/cells/{token}/stamp", cell_stamp),
    Route("/workspace/{token}/files", workspace_files),
    Route("/workspace/{token}/view", workspace_view, methods=["GET", "POST"]),
    Route("/workspace/{token}/add", workspace_add, methods=["POST"]),
    Route("/studio/{channel}/context", studio_context),
    Route("/studio/{channel}/commands", studio_commands),
    Route("/studio/{channel}/command", studio_command, methods=["POST"]),
    Route("/studio/{channel}/ask", studio_ask, methods=["POST"]),
    Route("/studio/{channel}/chat", studio_chat, methods=["GET", "POST"]),
    Route("/studio/{channel}/cluster_names", studio_cluster_names, methods=["GET"]),
    Route("/studio/{channel}/memory", studio_memory, methods=["POST"]),
    Route("/studio/{channel}/describe", studio_describe, methods=["POST"]),
    Route("/studio/{channel}/step/{step}", studio_step),
    Route("/studio/{channel}/overview", studio_overview),
    Route("/studio/{channel}/background_levels", studio_background_levels),
    Route("/studio/{channel}/background_images", studio_background_images, methods=["GET", "POST"]),
    Route("/studio/{channel}/segmentation_settings", studio_segmentation_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/segmentation_images", studio_segmentation_images, methods=["GET", "POST"]),
    Route("/studio/{channel}/segmentation_regions", studio_segmentation_regions),
    Route("/studio/{channel}/segment_preview", studio_segment_preview, methods=["POST"]),
    Route("/studio/{channel}/tissue_settings", studio_tissue_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/tissue_preview", studio_tissue_preview, methods=["POST"]),
    Route("/studio/{channel}/artifact_settings", studio_artifact_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/artifact_preview", studio_artifact_preview, methods=["POST"]),
    Route("/studio/{channel}/artifact_example", studio_artifact_example, methods=["POST"]),
    Route("/studio/{channel}/pixel_classes", studio_pixel_classes, methods=["POST"]),
    Route("/studio/{channel}/class_names", studio_class_names, methods=["POST"]),
    Route("/studio/{channel}/project_status", studio_project_status),
    Route("/studio/{channel}/correction", studio_correction),
    Route("/studio/{channel}/correction/{action}", studio_correction, methods=["POST"]),
    Route("/studio/{channel}/colours", studio_colours, methods=["GET", "POST"]),
    Route("/studio/{channel}/phenotype_presets", studio_phenotype_presets, methods=["GET", "POST"]),
    Route("/studio/{channel}/samples", studio_samples),
    Route("/studio/{channel}/samples/{action}", studio_samples, methods=["POST"]),
    Route("/studio/{channel}/training", studio_training),
    Route("/studio/{channel}/training/{action}", studio_training, methods=["POST"]),
    Route("/studio/{channel}/batch_checks", studio_batch_checks),
    Route("/studio/{channel}/leiden_settings", studio_leiden_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/clustering/{method}", studio_clustering_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/marker_choices", studio_marker_choices),
    Route("/studio/{channel}/consensus_settings", studio_consensus_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/mrml_settings", studio_mrml_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/subpheno_settings", studio_subpheno_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/phenotype_values", studio_phenotype_values),
    Route("/studio/{channel}/cluster_suggestions", studio_cluster_suggestions),
    Route("/studio/{channel}/figure", studio_figure, methods=["POST"]),
    Route("/studio/{channel}/saved", studio_saved),
    Route("/studio/{channel}/saved/{action}", studio_saved, methods=["POST"]),
    Route("/studio/{channel}/ai", studio_ai),
    Route("/studio/{channel}/ai/{action}", studio_ai, methods=["POST"]),
    Route("/studio/{channel}/neighbourhood_settings", studio_neighbourhood_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/tagging_settings", studio_tagging_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/folders/{step}", studio_step_folders, methods=["GET", "POST"]),
    Route("/studio/{channel}/tagging_preview", studio_tagging_preview, methods=["POST"]),
    Route("/studio/{channel}/interaction_settings", studio_interaction_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/tab8_results", studio_tab8_results),
    Route("/studio/{channel}/roi_markers", studio_roi_markers, methods=["GET", "POST"]),
    Route("/studio/{channel}/roi_images", studio_roi_images, methods=["GET", "POST"]),
    Route("/studio/{channel}/annotation_images", studio_annotation_images, methods=["GET", "POST"]),
    Route("/studio/{channel}/model_library", studio_model_library, methods=["GET", "POST"]),
    Route("/studio/{channel}/manual_reference", studio_manual_reference, methods=["GET", "POST"]),
    Route("/studio/{channel}/auto_reference", studio_auto_reference, methods=["GET", "POST"]),
    Route("/studio/{channel}/reference_features", studio_reference_features, methods=["GET", "POST"]),
    Route("/studio/{channel}/projects", studio_projects, methods=["GET", "POST"]),
    Route("/studio/{channel}/marker_regions", studio_marker_regions, methods=["GET", "POST"]),
    Route("/studio/{channel}/registration_pairs", studio_registration_pairs, methods=["GET", "POST"]),
    Route("/studio/{channel}/registration_markers", studio_registration_markers, methods=["GET", "POST"]),
    Route("/studio/{channel}/registration_rotation", studio_registration_rotation, methods=["POST"]),
    Route("/studio/{channel}/registration_quick_align", studio_registration_quick_align, methods=["POST"]),
    Route("/studio/{channel}/registration_figure", studio_registration_figure),
    Route("/studio/{channel}/registration_quality", studio_registration_quality),
    Route("/studio/{channel}/pieces_settings", studio_pieces_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/pieces_preview", studio_pieces_preview, methods=["POST"]),
    Route("/studio/{channel}/zone_rules", studio_zone_rules, methods=["GET", "POST"]),
    Route("/studio/{channel}/zones_preview", studio_zones_preview, methods=["POST"]),
    Route("/studio/{channel}/selection_options", studio_selection_options),
    Route("/studio/{channel}/settings", studio_settings, methods=["GET", "POST"]),
    Route("/studio/{channel}/jobs", studio_jobs),
    Route("/studio/{channel}/jobs/{job_id}", studio_jobs, methods=["GET", "POST"]),
    Route("/session/{session}", session_get),
    Route("/session/{session}/images", session_images),
    Route("/session/{session}/open", session_open, methods=["POST"]),
    Route("/session/{session}/add-image", session_add_image, methods=["POST"]),
    Route("/resolve/{token}/open", resolve_open, methods=["POST"]),
    Route("/resolve/{token}/thresholds", resolve_thresholds, methods=["POST"]),
    Route("/resolve/{token}/tile/{z:int}/{x:int}/{y:int}.png", resolve_tile),
    Route("/resolve/{token}/legend", resolve_legend),
    Route("/resolve/{token}/histogram", resolve_histogram),
    Route("/resolve/{token}/why", resolve_why),
    Route("/resolve/{token}/save", resolve_save, methods=["POST"]),
    Route("/resolve/{token}/panel", resolve_panel, methods=["GET", "POST"]),
    Route("/display/{token}", display_get),
    Route("/display/{token}/save", display_save, methods=["POST"]),
    Route("/display/{token}/sets", display_sets, methods=["GET", "POST"]),
]


def _window_files():
    """The viewer page itself, for a viewer opened in its own browser window."""
    from pathlib import Path

    from starlette.staticfiles import StaticFiles

    import re

    from starlette.responses import HTMLResponse

    folder = Path(__file__).resolve().parent / "frontend"
    files = StaticFiles(directory=str(folder), html=True)
    # A process serves the UI that belongs to its loaded Python routes. Editing
    # this checkout must not expose half an update to scientists using it.
    from hashlib import sha256
    assets = {str(p.relative_to(folder)): (p.read_bytes(), int(p.stat().st_mtime))
              for p in folder.rglob('*') if p.is_file() and p.suffix in ('.js','.css','.html')}

    def page():
        # Each script and stylesheet is asked for with its file's date: a changed file is a new address, so no
        # browser can keep using an old copy.
        html = assets['index.html'][0].decode()

        def stamped(match):
            stamp = assets.get(match.group(2),(b'',0))[1]
            return f'{match.group(1)}="{match.group(2)}?v={stamp}"'

        return re.sub(r'(src|href)="((?!https?:|//)[^"?]+\.(?:js|css))"', stamped, html)

    async def app(scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "http" and (path.endswith("/") or path.endswith("/index.html") or path in ("", "/")):
            await HTMLResponse(page(), headers={"Cache-Control": "no-store"})(scope, receive, send)
            return
        relative = path.split('/window/',1)[-1].lstrip('/')
        if scope['type']=='http' and relative in assets:
            raw,_ = assets[relative]
            etag='"'+sha256(raw).hexdigest()+'"'
            headers={"Cache-Control":"no-cache","ETag":etag}
            unchanged=dict(scope.get('headers',[])).get(b'if-none-match',b'').decode()==etag
            media='text/javascript' if relative.endswith('.js') else 'text/css' if relative.endswith('.css') else 'text/html'
            await Response(b'' if unchanged else raw,status_code=304 if unchanged else 200,media_type=media,headers=headers)(scope,receive,send)
            return
        # The page's code is checked with the server on every load (a cheap "not modified" when unchanged), so a
        # new version is used at once instead of an old copy the browser kept.
        async def with_revalidation(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message = {**message, "headers": headers + [(b"cache-control", b"no-cache")]}
            await send(message)

        await files(scope, receive, with_revalidation)

    return app


routes.append(Mount("/window", app=_window_files()))
from .annotation_recording import install as _record_annotation_routes, workflow_record
_record_annotation_routes(routes)
routes.append(Route('/studio/{channel}/annotation_workflow', workflow_record, methods=['GET', 'POST']))
app = Starlette(routes=routes)

MOUNT_PATH = "/spacemap-viewer/api"
