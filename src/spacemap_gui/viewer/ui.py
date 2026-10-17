"""Opening the viewer from any tab.

Every page has an "Open image viewer" button (drawn by ``app.py``). It opens a
large pop-up with a picker over every image the project holds -- raw scans,
registered images, ROIs -- and draws each with everything that belongs to it:
its markers, its cells coloured by any column, its annotations, and the Manual
Reference labelling tools. Tabs can also open it on a specific image with
:func:`open_viewer`.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
from pathlib import Path

import streamlit as st

from spacemap.viewer import catalog

from . import registry
from .component import spacemap_viewer

_PRESET_KEY = "_viewer_preset"
_OPEN_KEY = "_viewer_open"
_CHANNEL_DEFAULTS = {  # marker -> colour used when the viewer first opens an image
    "DAPI": "#4f7dff",
    "Pan-Cytokeratin": "#ff3b3b",
    "PanCK": "#ff3b3b",
    "CD45": "#00e05a",
    "E-cadherin": "#ff8c00",
    "CD3e": "#ffd400",
    "CD8": "#ff00ff",
    "CD4": "#00e5ff",
    "CD68": "#9b5de5",
}
_HE_CHANNELS = (("HE_Red", "#ff0000"), ("HE_Green", "#00ff00"), ("HE_Blue", "#0000ff"))
from spacemap.viewer.reference import REFERENCE_COLUMN as LABEL_COLUMN_PATTERN  # noqa: E402  (whole-word "ref")


# A viewer window asks the server to open images outside any Streamlit run, so
# the project it belongs to is handed over as a plain context (project_context())
# and used in place of the session while its layers are built.
_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("spacemap_viewer_context", default=None)


def project_context(context: str = "") -> dict:
    """What building layers needs from the session: project, folders, table, colours, tab."""
    root, mappings = _project()
    return {
        "root": str(root) if root else None,
        "mappings": mappings,
        "unified": unified_table_path(),
        "colors": _saved_phenotype_colors(),
        "tab": context,
    }


@contextlib.contextmanager
def using_context(context: dict):
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _project() -> tuple[Path | None, dict]:
    override = _CONTEXT.get()
    if override is not None:
        root = Path(override["root"]) if override.get("root") else None
        mappings = dict(override.get("mappings") or {})
        if root is not None:  # the folders as saved now (a step may have changed one since the project opened)
            try:
                from spacemap_gui.studio.settings import read_config

                mappings.update(read_config(root).get("directory_mappings") or {})
            except (OSError, ValueError):
                pass
        return root, mappings
    root = st.session_state.get("active_project_root")
    config = st.session_state.get("active_project") or {}
    return (Path(root) if root else None), (config.get("directory_mappings") or {})


def unified_table_path() -> str | None:
    """The unified per-cell table the tabs are using."""
    override = _CONTEXT.get()
    if override is not None:
        return override.get("unified")
    for dir_key, file_key in (
        ("global_pheno_dir", "global_pheno_file"),
        ("global_spatial_dir", "global_spatial_file"),
        ("global_visualization_dir", "global_visualization_file"),
    ):
        directory, name = st.session_state.get(dir_key), st.session_state.get(file_key)
        if directory and name and Path(directory, name).is_file():
            return str(Path(directory, name))
    root, mappings = _project()
    config = st.session_state.get("active_project") or {}
    name = config.get("unified_csv_filename")
    for folder in (mappings.get("bc_output_dir"), root and root / "outputs" / "batch_corrected"):
        if folder and name and Path(folder, name).is_file():
            return str(Path(folder, name))
    return None


def _saved_phenotype_colors() -> dict[str, str]:
    """The project's phenotype colours (Tab 9), keyed like Tab 9 keys them."""
    override = _CONTEXT.get()
    if override is not None:
        return dict(override.get("colors") or {})
    prefix = "tab9_actual_cell_color_val_"
    return {
        key[len(prefix) :]: value
        for key, value in st.session_state.items()
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, str)
    }


_WARMED: set[str] = set()


def _warm_markers(token: str) -> None:
    """Read the ROI's marker values in the background, so the first cell click is instant."""
    import threading

    from . import api

    if token in _WARMED:
        return
    _WARMED.add(token)
    layer = registry.resolve_object(token, "cells")
    if layer:
        threading.Thread(target=api.warm_markers, args=(layer,), name="spacemap-warm-markers", daemon=True).start()


def _cache_dir(root: Path) -> Path:
    return root / "outputs" / "viewer_cache" / "cell_labels"


@st.cache_data(show_spinner=False, ttl=60)
def _list_images(root: str, mappings_json: str) -> list[dict]:
    return [vars(entry) for entry in catalog.list_images(root, json.loads(mappings_json))]


def viewer_dir() -> Path | None:
    """Where the viewer keeps what it saves for the active project."""
    root, _ = _project()
    return root / "outputs" / "viewer" if root else None


def display_settings_path(image_path: str) -> Path | None:
    """The saved marker colours, ranges and names for one image (see spacemap.viewer.display)."""
    from spacemap.viewer.display import settings_path
    root, _ = _project()
    return settings_path(root, image_path) if root else None


def drawings_path(image_path: str) -> Path | None:
    """The shapes drawn on one image in the general viewer."""
    folder = viewer_dir()
    return folder / "shapes" / f"{Path(image_path).name}.geojson" if folder else None


def image_layer(
    path: str,
    *,
    name: str | None = None,
    channels: list[dict] | None = None,
    opacity: float = 1.0,
    visible: bool = True,
    layer_id: str = "image",
    save_display: bool = True,
) -> dict:
    """An image layer. With a project loaded, its marker settings are saved with the project."""
    token = registry.register(path)
    layer = {
        "id": layer_id,
        "type": "image",
        "name": name or Path(path).name,
        "token": token,
        "channels": channels or [],
        "opacity": opacity,
        "visible": visible,
        "remember": Path(path).name,
    }
    settings = display_settings_path(path) if save_display else None
    if settings is not None:
        layer["display_token"] = registry.register_object("display", path=str(settings), image=str(path))
    return layer


def shapes_layer(
    target: str | Path,
    image_path: str,
    *,
    name: str = "Shapes",
    editable: bool = True,
    sources: list[tuple[str, str]] | None = None,
    classes: list[dict] | None = None,
    default_class: str | None = None,
    layer_id: str = "shapes",
    visible: bool = True,
    regions: bool = False,
    names: list[str] | None = None,
    reference: str | Path | None = None,
    annotation_kind: str | None = None,
    tissue_label: str = "tissue",
) -> dict:
    """Shapes drawn on *image_path*, saved into the GeoJSON file *target*.

    *sources* are (label, path) GeoJSON files the user may import shapes from.
    *classes* ([{"name", "color"}]) offer fixed names for new shapes, such as
    "ROI" or "Background"; without them each shape is named freely.
    """
    image_token = registry.register(image_path)
    token = registry.register_object(
        "shapes",
        path=str(target),
        image=str(image_path),
        editable=editable,
        sources=[[label, str(path)] for label, path in (sources or [])],
        reference=str(reference) if reference else "",
        annotation_kind=annotation_kind,
        tissue_label=tissue_label,
    )
    return {
        "id": layer_id,
        "type": "shapes",
        "name": name,
        "token": token,
        "image_token": image_token,
        "editable": editable,
        "saved_file": str(Path(target).resolve()),
        "classes": classes or [],
        "default_class": default_class,
        "opacity": 1.0,
        "visible": visible,
        **({"regions": True, "names": list(names or [])} if regions else {}),
        **({"annotation_kind": annotation_kind} if annotation_kind else {}),
    }


def default_channels(channel_names: list[str]) -> list[dict]:
    """DAPI shown in blue; common markers get their usual colours but stay off."""
    presets = []
    for index, name in enumerate(channel_names):
        if name in _CHANNEL_DEFAULTS:
            presets.append({"index": index, "color": _CHANNEL_DEFAULTS[name], "visible": name == "DAPI"})
    if not any(p["visible"] for p in presets) and channel_names:
        presets.append({"index": 0, "color": "#4f7dff", "visible": True})
    return presets


def he_overlay(path: str, channel_names: list[str], *, visible: bool = False, opacity: float = 0.6) -> dict | None:
    """The registered H&E stored as three channels of a merged image, shown in true colour."""
    index = {name: i for i, name in enumerate(channel_names)}
    if not all(name in index for name, _ in _HE_CHANNELS):
        return None
    high = _he_high(path, index[_HE_CHANNELS[0][0]])
    channels = [
        {"index": index[name], "color": color, "visible": True, "low": 0, "high": high} for name, color in _HE_CHANNELS
    ]
    layer = image_layer(
        path,
        name="H&E (registered)",
        channels=channels,
        opacity=opacity,
        visible=visible,
        layer_id="he",
        save_display=False,
    )
    layer["remember"] = Path(path).name + (":he" if high == 255 else ":he16")  # a range remembered as 8-bit is not reused
    return layer


def _he_high(path: str, channel: int) -> int:
    """The H&E's white: 255 when stored as 8-bit values, 65535 when scaled to 16 bits (as SPACEMAP's merge writes it)."""
    from spacemap.viewer import render
    from spacemap.viewer.sources import image_info

    try:
        info = image_info(path)
        tile = render.raw_tile(path, info, render.max_zoom(info), 0, 0, channel)
        return 65535 if int(tile.max()) > 255 else 255
    except Exception:
        return 255


def resolve_files(root: Path, unified: str | None) -> dict | None:
    """RESOLVE's panel, threshold table and settings, where Tab 7 keeps them (None without a panel)."""
    import hashlib

    cache = Path(root) / "outputs" / "phenotyping" / "cache"
    panel = cache / "resolve_pheno_panel.json"
    if not panel.is_file() and unified:
        panel = cache / f"resolve_pheno_{hashlib.md5(str(unified).encode()).hexdigest()}.json"
    if not panel.is_file():
        return None
    settings = cache / "resolve_params.json"
    return {
        "panel": str(panel),
        "thresholds": str(cache / "resolve_thresholds_table.csv"),
        "settings": str(settings) if settings.is_file() else None,
    }


def layers_for_image(entry: dict, *, show_he: bool = False, context: str = "") -> tuple[list[dict], str | None]:
    """Every layer that belongs to one catalog image. Returns (layers, note).

    *context* is the tab the viewer was opened from; it picks what the cells
    are coloured by first (phenotypes in Tabs 7 and 9, neighbourhoods in Tab 8).
    """
    from spacemap import dataio
    from spacemap.viewer import cells
    from spacemap.viewer.sources import image_info, open_source

    root, mappings = _project()
    info = image_info(entry["path"])
    layers = [image_layer(entry["path"], name=entry["name"], channels=default_channels(info.channel_names))]
    # Only ordinary image browsing uses the quiet first-open default; explicit
    # registration/comparison views retain the layers their tools requested.
    layers[0]["default_dapi"] = True
    he = he_overlay(entry["path"], info.channel_names, visible=show_he)
    if he:
        layers.append(he)
    note = None
    if root is None:
        return layers, note
    bundle = catalog.bundle_for(root, mappings, catalog.ImageEntry(**entry), unified_table_path())
    from spacemap_gui.studio.annotation_names import hidden_files

    hidden = hidden_files(root, entry["path"])  # taken off this image's list in Studio's Layers (still on disk)
    try:  # named classes left from a deleted Classification
        from spacemap_gui.studio import annotation as studio_annotation
        from spacemap_gui.studio.annotation_names import stale_domains

        output = studio_annotation.all_folders(root, mappings)["output"]
        if stale_domains(root, entry["path"], output):
            hidden = hidden | {str(studio_annotation.domains(output, studio_annotation.sample_id(entry["path"])).resolve())}
    except (OSError, ValueError, KeyError):
        pass
    shown = []
    for label, path in bundle.annotations:
        resolved = str(Path(path).resolve())
        if resolved not in hidden and resolved not in {str(Path(p).resolve()) for _, p in shown}:
            shown.append((label, path))
    bundle.annotations = shown
    for number, (label, path) in enumerate(bundle.annotations[:16]):
        layers.append(
            {
                "id": f"ann{number}",
                "type": "annotations",
                "name": label,
                "token": registry.register_object("annotations", path=path, image=entry["path"]),
                "image_token": layers[0]["token"],
                "opacity": 0.7,
                "visible": context == "tab4" and number == 0,
            }
        )
    from spacemap_gui.studio import cell_layers

    try:  # the Segmentation step's own result first: re-segmenting shows the new cells
        from spacemap_gui.studio.settings import read_config

        mine = cell_layers.own(root, read_config(root), mappings, entry["path"], entry["sample"])
    except (OSError, ValueError, KeyError):
        mine = None
    if mine:
        bundle.segmentation = mine
        if not bundle.cell_tables:
            bundle.cell_tables = catalog.find_cell_tables(root, mappings, unified_table_path())

    def make_cells(segmentation, layer_id, own=(), only=False):
        # Built in the background the first time, so the image shows at once.
        note = None
        pyramid, ready = cells.prepare_label_pyramid(segmentation, info.width, info.height, _cache_dir(root))
        if not ready:
            note = "Preparing this ROI's cell outlines (first time only); the cells appear by themselves."
        elif cells.pyramid_cell_count(pyramid) == 0:
            note = (
                f"`{Path(segmentation).name}` has no numbered cells (no `label` property), "
                "so there are no cell outlines to show."
            )
        # This segmentation's own tables first. A combined table made before these cells numbers other cells:
        # it stays listed, named so, and is not the one coloured by.
        # Other tables (the combined table, spatial results) only when they hold these very cells: the same numbers at
        # the same places as the segmentation's own table (or, without one, as many cells). Others are not offered.
        from spacemap_gui.studio import cell_view

        reference = next((path for name, path in own if name == "This segmentation's cells"), None)
        count = cells.pyramid_cell_count(pyramid) if cells.pyramid_ready(pyramid) else None
        older, left_out = [], []
        for name, path in ([] if only else bundle.cell_tables):
            fits, reason = cell_view.matches(reference, path, entry["sample"], count)
            (older if fits else left_out).append((name, path) if fits else {"name": name, "reason": reason})
        tables = [{"id": f"t{i}", "name": name, "path": path} for i, (name, path) in enumerate(list(own) + older)]
        point_tables = [
            {"id": f"p{i}", "name": name, "path": path}
            for i, (name, path) in enumerate(catalog.find_point_tables(root))
        ]
        columns = []
        for table in tables:
            try:
                columns.append(dataio.columns(table["path"]))
            except Exception:
                columns.append([])
        table_index, color_by = catalog.suggest_colouring(context, [(t["name"], c) for t, c in zip(tables, columns)])
        if own:  # the cells' own table, outlines only until a column is chosen
            table_index, color_by = 0, ""
        label_columns = [c for c in (columns[0] if columns else []) if LABEL_COLUMN_PATTERN.search(c)]
        # phenotype_ref1 first: the first entry is the viewer's default column for manual labels.
        label_columns = sorted(
            set(label_columns) | {"phenotype_ref1"}, key=lambda c: (c != "phenotype_ref1", c.casefold())
        )
        token = registry.register_object(
            "cells",
            image=entry["path"],
            pyramid=str(pyramid),
            roi=entry["sample"],
            segmentation=segmentation,
            background_image=catalog.background_image_for(root, mappings, catalog.ImageEntry(**entry)),
            tables={t["id"]: t["path"] for t in tables + point_tables},
            reference_table=reference,
            colors=_saved_phenotype_colors(),
        )
        _warm_markers(token)
        cells_layer = {
            "id": layer_id,
            "type": "cells",
            # Named by the folder they come from (segmentation_new, say): the file names are the same in every run.
            "name": f"Cells ({(Path(segmentation).parent.parent if Path(segmentation).parent.name == 'geojson' else Path(segmentation).parent).name})",
            "source": str(Path(segmentation).relative_to(root)) if root in Path(segmentation).parents else str(segmentation),
            "token": token,
            "image_token": layers[0]["token"],
            "roi": entry["sample"],  # the image's name in a combined table (batch_id)
            "mode": "both" if color_by else "outline",
            "tables": [{"id": t["id"], "name": t["name"]} for t in tables],
            "density_tables": [{"id": t["id"], "name": t["name"]} for t in tables + point_tables],
            "table": tables[table_index]["id"] if tables else "",
            "color_by": color_by,
            "label_columns": label_columns,
            "opacity": 0.9,
            "visible": True,
            "pending": not ready,
        }
        cells_layer["has_background_image"] = bool(registry.resolve_object(token, "cells").get("background_image"))
        cells_layer["hidden_tables"] = left_out
        unified = unified_table_path()
        # Live RESOLVE reads these cells' own raw markers when they have their own table: a combined table's
        # numbers may belong to an earlier segmentation.
        mine = next((path for name, path in own if name == "This segmentation's cells"), None)
        files = resolve_files(root, unified) if (unified or mine) else None
        if files:
            # Live RESOLVE for this ROI (thresholds on raw markers, phenotypes as Tab 7 assigns them).
            cells_layer["resolve"] = registry.register_object(
                "resolve", cells=token, table=mine or unified, roi=entry["sample"], colors=_saved_phenotype_colors(), **files
            )
        return cells_layer, note

    own_tables = []
    if mine:
        try:
            own_tables = cell_layers.own_tables(root, read_config(root), mappings, Path(entry["path"]).name, entry["sample"], _cache_dir(root))
        except (OSError, ValueError, KeyError):
            own_tables = []
    from spacemap_gui.studio import cell_data

    data = cell_data.records(root, entry["path"])
    # Cell data shown on the image's own cells: its table joins theirs (after their own, before an older combined one).
    linked = [("Data: " + item["name"], item["table"]) for item in data if item["show"] == "cells"]
    if bundle.segmentation:
        cells_layer, made_note = make_cells(bundle.segmentation, "cells", list(own_tables) + linked)
        note = made_note or note
        layers.append(cells_layer)
    # Cell data drawn as circles or Voronoi cells: each its own cell layer, coloured by its own table.
    for item in data:
        if item["show"] == "cells" or not item.get("geojson") or not Path(item["geojson"]).is_file():
            continue
        extra, _ = make_cells(item["geojson"], "data-" + item["id"], [("Data: " + item["name"], item["table"])], only=True)
        extra.update(added=True, data_layer=item["id"], data_text=cell_data.describe(item), visible=True, data_show=item["show"],
                     data_size=item["choice"].get("size_um"), data_reach=item["choice"].get("reach_um"), data_within=item.get("within_key", ""),
                     name=("Circles" if item["show"] == "circles" else "Voronoi") + f" ({item['name']})",
                     mode="fill" if item["show"] == "circles" else "outline", color_by="")
        extra.pop("resolve", None)
        layers.append(extra)
    # Segmentations imported in Layers (another run, a colleague's outlines): each its own layer, removable.
    for path in cell_layers.imported(root, entry["path"]):
        if bundle.segmentation and Path(path).resolve() == Path(bundle.segmentation).resolve():
            continue
        # Its own cell table (dropped with it, or beside it by name) is its data, as Segmentation's table is for its cells.
        table = cell_layers.table_of(root, entry["path"], path)
        own = [("Its cells' table", cell_layers._with_roi(table, entry["sample"], _cache_dir(root)))] if table else []
        extra, _ = make_cells(path, "cells-" + __import__("hashlib").sha256(path.encode()).hexdigest()[:12], own, only=bool(own))
        extra.update(added=True, imported_cells=True, visible=False, mode="outline", color_by="")
        extra.pop("resolve", None)  # live RESOLVE stays with the main cells
        layers.append(extra)
    if not bundle.segmentation and entry["group"] == "ROIs":
        note = "No cell segmentation found for this ROI yet (Tab 5)."
    drawings = drawings_path(entry["path"])
    if drawings is not None:
        layers.append(
            shapes_layer(drawings, entry["path"], name="Drawings", layer_id="drawings", sources=bundle.annotations)
        )
    return layers, note


_SESSION_KEY = "_viewer_session"  # id of this browser session's viewer-window channel
_LAST_KEY = "_viewer_last"  # the last view shown, for Restore and the viewer window
_HANDLED_KEY = "_viewer_handled_event"
_MINIMIZED_KEY = "_viewer_minimized"


def _session_id() -> str:
    import secrets

    session = st.session_state.get(_SESSION_KEY)
    if not session:
        session = st.session_state[_SESSION_KEY] = secrets.token_hex(16)
    return session


def _workspace_token(image_path: str | None) -> str | None:
    """Lets the viewer add GeoJSON files: the project's, or ones saved into outputs/viewer/imported."""
    root, _ = _project()
    folder = viewer_dir()
    if root is None or folder is None or not image_path:
        return None
    return registry.register_object(
        "workspace", root=str(root), imports=str(folder / "imported"), image=str(image_path)
    )


def _spec(title: str, layers: list[dict], note: str | None = None) -> dict:
    image = next((layer for layer in layers if layer.get("type") == "image"), None)
    spec = {
        "id": _spec_id(title, layers),
        "title": title,
        "layers": layers,
        "note": note or "",
        "session": _session_id(),
        "height": 780,
    }
    workspace = _workspace_token(registry.resolve(image["token"]) if image else None)
    if workspace:
        spec["workspace"] = workspace
    return spec


def _publish(spec: dict, context: str = "") -> None:
    """Send the view to this session's viewer window (if one is open, it switches to it).

    The project context goes with it, so the window can open the project's other
    images itself (window_open_image).
    """
    from .component import api_location

    registry.publish(spec["session"], {**api_location(), **spec, "standalone": True}, project=project_context(context))
    st.session_state[_LAST_KEY] = spec


def _extra_codex_images(project: dict) -> list:
    """Registration's extra CODEX folders (CODEX (2), (3), ...), so their images can be opened too."""
    from spacemap.background_subtraction_utils import is_supported_image_file
    from spacemap_gui.studio import registration
    from spacemap_gui.studio.settings import read_config

    root = project.get("root")
    if not root:
        return []
    try:
        folders = registration.extra_folders(root, read_config(Path(root)))
    except (OSError, ValueError):
        return []
    entries = []
    for number, folder in enumerate(folders, start=2):
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and is_supported_image_file(path.name):
                    entries.append(catalog.ImageEntry(group=f"Raw CODEX ({number})", name=path.name, path=str(path), sample=catalog.sample_of(path.name)))
    return entries


def _background_images(project: dict, known: set) -> list:
    """The Background step's images folder (any folder: registered, ROIs, raw QPTIFF), so its images can be opened."""
    from spacemap_gui.studio import background
    from spacemap_gui.studio.settings import read_config

    root = project.get("root")
    if not root:
        return []
    try:
        config = read_config(Path(root))
        folder = background.all_folders(root, current_mappings(project), config)["images"]
    except (OSError, ValueError):
        return []
    if not folder.is_dir():
        return []
    return [catalog.ImageEntry(group="Background images", name=path.name, path=str(path), sample=catalog.sample_of(path.name))
            for path in sorted(folder.iterdir())
            if path.is_file() and background._is_image(path.name) and str(path.resolve()) not in known]


def current_mappings(project: dict) -> dict:
    """The session's folders with the project file's current ones on top (a folder changed in Studio counts at once)."""
    from spacemap_gui.studio.settings import read_config

    mappings = dict(project.get("mappings") or {})
    root = project.get("root")
    if root:
        try:
            saved = read_config(Path(root)).get("directory_mappings") or {}
        except (OSError, ValueError):
            saved = {}
        mappings.update({key: str(value).replace("${PROJECT_ROOT}", str(root)) for key, value in saved.items() if value})
    return mappings


def window_images(session: str) -> list[dict]:
    """The images a viewer window can switch to (no paths: ids, groups and names)."""
    project = registry.session_project(session)
    if not project:
        return []
    entries = catalog.list_images(project["root"], current_mappings(project)) if project.get("root") else []
    entries += catalog.added_images(project.get("root"))
    entries += _extra_codex_images(project)
    entries += _background_images(project, {str(Path(e.path).resolve()) for e in entries})
    # One name in two folders (a new registration folder, say): name each by its folder, so both can be opened.
    from collections import Counter
    from dataclasses import replace

    twice = Counter((e.group, e.name) for e in entries)
    root = Path(project.get("root") or ".")
    for index, entry in enumerate(entries):
        if twice[(entry.group, entry.name)] > 1:
            folder = Path(entry.path).parent
            try:
                folder = folder.relative_to(root)
            except ValueError:
                pass
            entries[index] = replace(entry, group=f"{entry.group} ({folder})")
    registry.session_remember_images(session, [vars(e) for e in entries])
    return [{"id": i, "group": e.group, "name": e.name, "sample": e.sample} for i, e in enumerate(entries)]


def window_add_image(session: str, path: str) -> dict:
    """Add an image from outside the project, from a viewer window: the new list and the added one's id."""
    project = registry.session_project(session) or {}
    entry = catalog.add_image(project.get("root"), path)
    images = window_images(session)
    added = next(i for i in reversed(images) if (i["group"], i["name"]) == (entry.group, entry.name))
    return {"images": images, "added": added["id"]}


def window_image_spec(session: str, image_id: int) -> dict:
    """Build a checked image view without publishing it until its request is valid."""
    from .component import api_location

    project = registry.session_project(session)
    entries = registry.session_images(session)
    if not project or not (0 <= image_id < len(entries)):
        raise ValueError("That image is no longer available. Reopen the window from SPACEMAP.")
    entry = entries[image_id]
    with using_context(project):
        layers, note = layers_for_image(entry, context=project.get("tab", ""))
        title = f"{entry['sample']}  |  {entry['group']}"
        spec = {
            "id": _spec_id(title, layers),
            "title": title,
            "layers": layers,
            "note": note or "",
            "session": session,
            "height": 780,
        }
        workspace = _workspace_token(entry["path"])
        if workspace:
            spec["workspace"] = workspace
    return {**api_location(), **spec, "standalone": True}


def window_open_image(session: str, image_id: int) -> dict:
    """Open one checked project image in the linked viewer window."""
    full = window_image_spec(session, image_id)
    version = registry.publish(session, full, project=registry.session_project(session))
    return {"version": version, "spec": full}


def viewer_window_open() -> bool:
    session = st.session_state.get(_SESSION_KEY)
    return bool(session) and registry.window_open(session)


def open_viewer(title: str, layers: list[dict]) -> None:
    """Show specific layers (for tab-specific buttons).

    With a viewer window open, the window switches to them and the page carries
    on; otherwise the pop-up opens.
    """
    if viewer_window_open():
        _publish(_spec(title, layers))
        st.toast(f"Shown in the viewer window: {title}")
        return
    st.session_state[_PRESET_KEY] = {"title": title, "layers": layers}
    st.session_state[_OPEN_KEY] = True
    st.session_state.pop(_MINIMIZED_KEY, None)
    # Buttons inside a fragment rerun only the fragment; the pop-up is drawn by
    # the page, so ask for a full rerun.
    st.rerun(scope="app")


def _spec_id(title: str, layers: list[dict]) -> str:
    material = json.dumps([title, [(l["id"], l["token"]) for l in layers]], sort_keys=True)
    return hashlib.sha1(material.encode()).hexdigest()[:16]


@st.dialog("Image viewer", width="large")
def _viewer_dialog(context: str = "") -> None:
    command = st.session_state.get("_viewer_command_result")
    if command:
        spec = {**command["spec"], "_studio_version": command["version"]}
        spacemap_viewer(spec, key="spacemap_command_viewer")
        return
    preset = st.session_state.get(_PRESET_KEY)
    note = None
    if preset:
        title, layers = preset["title"], preset["layers"]
        note = preset.get("note")
        st.caption(title)
    else:
        root, mappings = _project()
        found = _list_images(str(root), json.dumps(mappings, sort_keys=True, default=str)) if root else []
        choice_key = f"viewer_image_choice_{context or 'any'}"
        _render_add_image(root, choice_key, expanded=not found and not catalog.added_images(root))
        images = found + [vars(e) for e in catalog.added_images(root)]  # read after the form: may hold a new one
        if not images:
            st.info(
                "No images yet: add one with the box above."
                if root is None
                else "No images found in this project yet; you can add one with the box above."
            )
            return
        labels = [f"{image['group']}  |  {image['name']}" for image in images]
        if st.session_state.get(choice_key) not in labels:
            st.session_state.pop(choice_key, None)
        wanted = catalog.start_group(context)
        start = next((i for i, image in enumerate(images) if image["group"] == wanted), 0)
        placed = {} if choice_key in st.session_state else {"index": start}  # set by code or by you: no default
        choice = st.selectbox(
            "Image",
            labels,
            key=choice_key,
            **placed,
            help="Every image the project holds: raw scans, registered images, ROIs, and "
            "any image you added from elsewhere. Cells, annotations and phenotypes are "
            "drawn on ROIs.",
        )
        entry = images[labels.index(choice)]
        if entry["group"] == catalog.ADDED_GROUP:
            if st.button(
                "Remove from the list",
                key="viewer_remove_added_image_btn",
                help="Takes this image off the list. The file itself is not touched.",
            ):
                catalog.remove_image(root, entry["path"])
                st.session_state.pop(choice_key, None)
                st.session_state[_OPEN_KEY] = True
                st.rerun()
        try:
            layers, note = layers_for_image(entry, context=context)
        except Exception as exc:
            st.error(f"Could not open {entry['name']}: {exc}")
            return
        title = f"{entry['sample']}  |  {entry['group']}"
    if note:
        st.caption(note)
    # The browser's own channel id, reported by the viewer on an earlier update:
    # views published under it reach a viewer window from this browser, even
    # after SPACEMAP restarts. Taken before the view is published, so no extra
    # page update is needed (a fragment-only rerun is not allowed when the
    # pop-up opens during a full page update).
    _adopt_browser_channel(st.session_state.get("spacemap_viewer_component"))
    spec = _spec(title, layers, note)
    _publish(spec, context)
    event = spacemap_viewer(spec, key="spacemap_viewer_component")
    if _adopt_browser_channel(event):
        _publish(_spec(title, layers, note), context)  # this update's view, under the browser's channel
    if isinstance(event, dict) and event.get("event") in ("minimize", "window"):
        if st.session_state.get(_HANDLED_KEY) != event.get("n"):
            st.session_state[_HANDLED_KEY] = event.get("n")
            st.session_state[_MINIMIZED_KEY] = True
            st.rerun(scope="app")  # closes the pop-up; Restore brings the same view back


def _render_add_image(root: Path | None, choice_key: str, expanded: bool) -> None:
    """Add an image from anywhere (a scan elsewhere on the cluster) to this project's list."""
    with st.expander("Add an image from outside the project", expanded=expanded):
        st.caption(
            "Paste the full path of a QPTIFF, OME-TIFF, TIFF or SVS image. It is added to the list "
            "(remembered with the project); the file is not copied or changed."
        )
        path = st.text_input("Image path", key="viewer_add_image_path", placeholder="/path/to/image.qptiff")
        if st.button("Add to the list", key="viewer_add_image_btn", disabled=not str(path or "").strip()):
            try:
                entry = catalog.add_image(root, path)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state[choice_key] = f"{entry.group}  |  {entry.name}"
            st.success(f"Added {entry.name}.")


def _adopt_browser_channel(event) -> bool:
    """Use the channel id the viewer reported for this browser. True when it changed."""
    browser = str(event.get("browser") or "") if isinstance(event, dict) else ""
    if re.fullmatch(r"[0-9a-f]{32}", browser) and st.session_state.get(_SESSION_KEY) != browser:
        st.session_state[_SESSION_KEY] = browser
        return True
    return False


def studio_project_choice() -> str | None:
    """The project folder Studio opened since this page last ran (the Project step's list), once."""
    if _SESSION_KEY not in st.session_state:
        return None
    return registry.take_choice(_session_id())


def apply_studio_table_choice() -> None:
    """Adopt Studio's input in classic before any file-selection widgets are rendered."""
    if _SESSION_KEY not in st.session_state:
        return
    chosen = registry.take_table_choice(_session_id())
    root = st.session_state.get("active_project_root")
    if not chosen or not root or Path(chosen["root"]).resolve() != Path(root).resolve():
        return
    path = Path(chosen["table"])
    st.session_state.update(global_pheno_dir=str(path.parent), global_pheno_file=path.name,
                            global_pheno_file_cached=path.name, global_pheno_dir_cached=str(path.parent),
                            global_pheno_dir_user_override=True, proj_unified_csv_filename=path.name,
                            global_pheno_last_synced_bc=st.session_state.get("bc_last_run_file"))
    project = st.session_state.get("active_project")
    if isinstance(project, dict):
        project.setdefault("directory_mappings", {})["global_pheno_dir"] = str(path.parent)
        project["unified_csv_filename"] = path.name
        project.setdefault("tab_configurations", {}).setdefault("tab7_phenotyping", {}).update(
            global_pheno_file=path.name, global_pheno_dir_user_override=True)


def studio_link(context: str) -> str | None:
    """Link the current project without opening an image or loading pipeline code."""
    from urllib.parse import quote
    from . import api
    from .component import api_location

    if not api.MOUNTED:
        return None
    channel = _session_id()
    project = project_context(context)
    previous_project = registry.session_project(channel) or {}
    view = registry.session_view(channel, mark_seen=False)
    same_project = previous_project.get("root") == project.get("root")
    if view is None or not same_project:
        spec = {
            **api_location(),
            "id": "studio-empty-" + channel,
            "title": "Choose an image",
            "layers": [],
            "session": channel,
            "standalone": True,
        }
    else:
        spec = view["spec"]
    registry.publish(channel, spec, project=project)
    return api_location()["api"] + "/window/index.html?studio=1&session=" + quote(channel) + "&step=" + quote(context)


def render_viewer_launcher(context: str, studio_area=None, viewer_area=None) -> None:
    """The SPACEMAP Studio and Image viewer buttons every page shows.

    The app puts them in the top bar next to the AI Assistant (*studio_area*
    and *viewer_area* are its columns); without them they go where this is called.
    """
    from contextlib import nullcontext

    last = st.session_state.get(_LAST_KEY)
    studio_url = studio_link(context)
    with studio_area or nullcontext():
        if studio_url:
            st.link_button(
                "SPACEMAP Studio",
                studio_url,
                icon=":material/dashboard:",
                width="stretch",
                help="Steps, images and cell review together, with links to the classic tools.",
            )
    with viewer_area or nullcontext():
        if st.button(
            "Image viewer",
            key=f"{context}_open_image_viewer_btn",
            icon=":material/image:",
            width="stretch",
            help="Browse the project's images with their cells, annotations and phenotypes. "
            "Zoom, colour cells by any column, label cells, and save views as images.",
        ):
            st.session_state.pop("_viewer_command_result", None)
            st.session_state.pop(_PRESET_KEY, None)
            st.session_state.pop(_MINIMIZED_KEY, None)
            st.session_state[_OPEN_KEY] = True
    # Restore and the window note go below the top bar (where this is called): in the
    # button's column they made it taller, so its button no longer lined up with the others.
    with nullcontext():
        if last and st.session_state.get(_MINIMIZED_KEY) and not viewer_window_open():
            if st.button(
                "Restore image viewer",
                key=f"{context}_restore_image_viewer_btn",
                width="stretch",
                help=f"Reopen the viewer as you left it: {last['title']}.",
            ):
                st.session_state[_PRESET_KEY] = {
                    "title": last["title"],
                    "layers": last["layers"],
                    "note": last.get("note"),
                }
                st.session_state.pop(_MINIMIZED_KEY, None)
                st.session_state[_OPEN_KEY] = True
        if viewer_window_open():
            st.caption("Viewer window open: images opened from the tabs show there.")
    if st.session_state.pop(_OPEN_KEY, False):
        _viewer_dialog(context)
