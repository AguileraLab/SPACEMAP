"""A model library shared by projects: trained models are copied in (Share) and copied into another project (Import).

The library is one folder, by default ``<SPACEMAP home>/models/library``; it can be any folder, such as a lab drive,
so other people and computers use the same models. Its layout is ``<library>/<kind>/<entry>/`` with the model's
files and ``model.json`` describing it (what it is, where and when it was trained, on what, and how well it scored).

Kinds:

* ``cellpose``: a Cellpose model trained in Segmentation. Importing puts it with the project's trained Cellpose
  models, where Segmentation lists it.
* ``artifact``: an artifact model version (blood clots, image quality, ...). Importing adds it to the project's versions
  of that artifact, where Artifacts lists it.
* ``phenotyping``: a phenotyping reference model (cell suggestions). Importing makes it usable on other cell tables
  that have its markers.

Nothing in the library is changed by a project, and nothing in a project is changed by sharing.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

KINDS = {"cellpose": "Segmentation (Cellpose)", "artifact": "Artifacts", "phenotyping": "Phenotyping"}
SETTINGS_FILE = "model_library.json"
ENVIRONMENT = "SPACEMAP_MODEL_LIBRARY"


def _settings_path():
    from spacemap.userpaths import config_dir

    return Path(config_dir()) / SETTINGS_FILE


def location():
    """The library folder: the environment variable, else the chosen folder, else the default."""
    from spacemap.userpaths import models_dir

    if os.environ.get(ENVIRONMENT):
        return Path(os.environ[ENVIRONMENT]).expanduser()
    try:
        chosen = json.loads(_settings_path().read_text()).get("path")
    except (OSError, ValueError):
        chosen = None
    return Path(chosen).expanduser() if chosen else Path(models_dir()) / "library"


def set_location(path):
    """Use another folder as the library (made if its parent exists)."""
    if not isinstance(path, str) or not path.strip() or len(path) > 1024:
        raise ValueError("Type the library folder's full path.")
    target = Path(path.strip()).expanduser()
    if not target.is_absolute():
        raise ValueError("Type the full path of the library folder (starting with /).")
    if not target.is_dir():
        if not target.parent.is_dir():
            raise ValueError(f"{target.parent} is not a folder.")
        target.mkdir()
    settings = _settings_path()
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"path": str(target)}, indent=2))
    return target


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")[:120] or "model"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _copy(source, target):
    """Copy a file whole, then move it into place, so a reader never sees half a model."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name("." + target.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, target)
    from spacemap.sharing import share
    share(target.parent)
    share(target)  # a shared library folder: the group can read what one member adds


# ---------------------------------------------------------------- the project's models
def _artifact_parts():
    from . import artifact_learning

    return list(artifact_learning.NAMES)


def project_models(root):
    """The models trained in (or imported into) this project that can be shared."""
    from . import artifact_learning, reference_models, training

    found = []
    try:
        for model in training.examples(root)["models"]:
            found.append({"kind": "cellpose", "id": model["name"], "name": model["name"], "path": model["path"],
                          "created": model.get("created") or model.get("trained") or "",
                          "details": {k: model[k] for k in ("base", "scores", "examples", "imported_from") if k in model}})
    except Exception:  # no Cellpose training in this project
        pass
    for part in _artifact_parts():
        home = artifact_learning.folder(root, part)
        for record in _read(home / "versions.json", []):
            if record.get("deleted_at") or not (home / (record["id"] + ".joblib")).is_file():
                continue
            found.append({"kind": "artifact", "id": record["id"], "name": record.get("name") or record["id"], "part": part,
                          "created": record.get("created", ""), "details": {
                              "artifact": artifact_learning.NAMES.get(part, part), "images": record.get("images"),
                              "samples": record.get("samples"), "markers": record.get("markers"), "he_backup": bool(record.get("he")),
                              "imported_from": record.get("imported_from")}})
    directory = reference_models.directory()
    tables = _project_tables(root)
    if directory.is_dir():
        for meta_path in sorted(directory.glob("*.json")):
            meta = _read(meta_path, None)
            if not meta or not (meta.get("table_path") in tables or meta.get("imported")):
                continue
            found.append({"kind": "phenotyping", "id": meta_path.stem, "name": f"{meta.get('method', 'Model')} · {meta.get('column', '')}",
                          "created": meta.get("trained", ""), "details": {
                              "table": meta.get("table"), "markers": meta.get("markers"), "examples": meta.get("examples"),
                              "held_out_balanced_accuracy": meta.get("held_out_balanced_accuracy"),
                              "marker_importance": meta.get("marker_importance"), "adaptable": bool(meta.get("adaptable")),
                              "imported_from": meta.get("imported_from")}})
    return found


def _project_tables(root):
    """Cell tables belonging to this project (phenotyping models are recorded by their table's path)."""
    root = Path(root).resolve()
    tables = set()
    for folder in ("outputs", "data"):
        base = root / folder
        if base.is_dir():
            for pattern in ("*.csv", "*.parquet"):
                tables.update(str(path.resolve()) for path in base.rglob(pattern))
    return tables


# ---------------------------------------------------------------- the library
def entries():
    """Every model in the library, newest first."""
    library = location()
    found = []
    for kind in KINDS:
        base = library / kind
        if not base.is_dir():
            continue
        for manifest in base.glob("*/model.json"):
            meta = _read(manifest, None)
            if meta and meta.get("kind") == kind:
                found.append({**meta, "entry": manifest.parent.name})
    return sorted(found, key=lambda item: item.get("shared", ""), reverse=True)


def share(root, kind, identifier, project_name=""):
    """Copy one of the project's models into the library, with a description. Returns the library entry."""
    from . import artifact_learning, reference_models

    model = next((m for m in project_models(root) if m["kind"] == kind and m["id"] == identifier), None)
    if model is None:
        raise ValueError("That model is not in this project.")
    entry = _safe(kind + "_" + identifier)
    target = location() / kind / entry
    if (target / "model.json").is_file():
        raise ValueError("This model is already in the library.")
    files = []
    if kind == "cellpose":
        source = Path(model["path"])
        _copy(source, target / source.name)
        files.append(source.name)
        extra = source.with_name(source.name + ".json")
        if extra.is_file():
            _copy(extra, target / extra.name)
            files.append(extra.name)
        record = _read(extra, {})
    elif kind == "artifact":
        home = artifact_learning.folder(root, model["part"])
        source = home / (identifier + ".joblib")
        _copy(source, target / source.name)
        files.append(source.name)
        record = next(r for r in _read(home / "versions.json", []) if r["id"] == identifier)
    else:
        source = reference_models.model_path(identifier)
        _copy(source, target / source.name)
        files.append(source.name)
        record = {k: v for k, v in _read(source.with_suffix(".json"), {}).items() if k not in ("table_path",)}
    manifest = {"kind": kind, "kind_name": KINDS[kind], "id": identifier, "name": model["name"], "part": model.get("part"),
                "created": model.get("created", ""), "shared": _now(), "source_project": project_name or Path(root).name,
                "details": model.get("details", {}), "record": record, "files": files}
    (target / "model.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {**manifest, "entry": entry}


def import_model(root, kind, entry):
    """Copy a library model into this project, where its step lists it. Returns what was added."""
    from . import artifact_learning, reference_models, training
    from .settings import project_lock

    if kind not in KINDS or not isinstance(entry, str) or "/" in entry or entry.startswith("."):
        raise ValueError("Choose a model from the library.")
    source = location() / kind / entry
    manifest = _read(source / "model.json", None)
    if not manifest or manifest.get("kind") != kind:
        raise ValueError("That library entry is missing or damaged.")
    origin = {"library": str(location()), "entry": entry, "project": manifest.get("source_project"), "shared": manifest.get("shared"),
              "imported": _now()}
    if kind == "cellpose":
        _, models = training.folders(root)
        name = manifest["files"][0]
        target = models / "models" / name
        if target.exists():
            raise ValueError(f"This project already has a Cellpose model named {name}.")
        _copy(source / name, target)
        info = {**(manifest.get("record") or {}), "imported_from": origin}
        target.with_name(name + ".json").write_text(json.dumps(info, indent=2, default=str))
        return {"kind": kind, "name": name, "path": str(target)}
    if kind == "artifact":
        part = manifest.get("part")
        if part not in _artifact_parts():
            raise ValueError("This artifact model is for an artifact this version of SPACEMAP does not know.")
        home = artifact_learning.folder(root, part)
        with project_lock(root):
            versions = _read(home / "versions.json", [])
            if any(v["id"] == manifest["id"] for v in versions):
                raise ValueError("This project already has that artifact model.")
            _copy(source / (manifest["id"] + ".joblib"), home / (manifest["id"] + ".joblib"))
            record = {**manifest["record"], "imported_from": origin}
            (home / "versions.json").write_text(json.dumps([*versions, record], indent=2, default=str))
        return {"kind": kind, "name": manifest["name"], "part": part}
    # Phenotyping: usable on any cell table with its markers.
    model_id = manifest["id"]
    target = reference_models.model_path(model_id)
    meta_path = target.with_suffix(".json")
    if target.exists() and _read(meta_path, {}).get("imported"):
        raise ValueError("This phenotyping model is already imported.")
    if not target.exists():
        _copy(source / (model_id + ".joblib"), target)
    meta = {**(_read(meta_path, None) or manifest.get("record") or {}), "imported": True, "imported_from": origin}
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return {"kind": kind, "name": manifest["name"]}
