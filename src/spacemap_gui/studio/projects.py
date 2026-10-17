"""The Project step's projects: the list classic SPACEMAP keeps, opening one from Studio, a new project, and its notes.

Opening a project in Studio switches Studio at once; classic SPACEMAP loads the same project the next time its page
runs (``take_choice``), saving the one it had open first, as its own project list does. Projects are the folders
classic Tab 0 knows (its projects index); a folder typed by hand is added to that list when it holds a project.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

NOTES = {"description": "proj_description", "pi": "proj_pi", "institution": "proj_institution"}


def _saved(root):
    from spacemap_gui.persistence import project_config_path

    path = project_config_path(root)
    if path is None:
        return None
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def known(current=None):
    """Every project in classic SPACEMAP's list, the latest saved first: name, folder, last saved, whether it is there."""
    from spacemap_gui.persistence import load_projects_index

    here = str(Path(current).resolve()) if current else None
    found = []
    for name, root in sorted(load_projects_index().items(), key=lambda item: item[0].lower()):
        saved = _saved(root) if root else None
        found.append({"name": name, "folder": str(root), "saved": saved, "missing": saved is None,
                      "open": bool(here and root and str(Path(root).resolve()) == here)})
    found.sort(key=lambda item: item["saved"] or "", reverse=True)  # the latest worked on first, missing folders last
    return found


def _context(root):
    """What Studio needs to work in *root* (folders, combined cell table, colours), read from its settings file."""
    from spacemap_gui import persistence
    from spacemap_gui.project_bundle import localize

    from . import colours

    config = persistence.load_project_config(str(root))
    if not config:
        raise ValueError(f"No project settings file in {root}.")
    mappings = localize(config.get("directory_mappings") or {}, root)
    pheno = localize((config.get("tab_configurations") or {}).get("tab7_phenotyping") or {}, root)
    unified = None
    names = [pheno.get("global_pheno_file"), config.get("unified_csv_filename")]
    folders = [mappings.get("global_pheno_dir"), pheno.get("global_pheno_dir"), pheno.get("global_pheno_dir_last_bc"), mappings.get("bc_output_dir"),
               str(Path(root) / "outputs" / "batch_corrected")]
    for name in filter(None, names):
        hit = next((Path(folder, name) for folder in folders if folder and Path(folder, name).is_file()), None)
        if hit is not None:
            unified = str(hit)
            break
    return {"root": str(root), "mappings": mappings, "unified": unified,
            "colors": colours.saved(config), "tab": "tab0"}


def open_project(channel, *, name=None, folder=None):
    """Make a project (by its name in the list, or a folder) the one this Studio window works in."""
    from spacemap_gui import persistence
    from spacemap_gui.viewer import registry

    view = registry.session_view(channel, mark_seen=False)
    if view is None:
        raise ValueError("Open Studio from SPACEMAP first.")
    if name is not None:
        index = persistence.load_projects_index()
        if name not in index:
            raise ValueError("That project is not in the list.")
        root = Path(index[name])
        if persistence.project_config_path(root) is None:
            moved = persistence.find_moved_project(name, index)
            if moved is None:
                raise ValueError(f"The folder of {name} was not found: {root}. Open it from its new folder.")
            root = moved
    else:
        if not isinstance(folder, str) or not folder.strip() or len(folder) > 4096:
            raise ValueError("Type the project folder's full path.")
        root = Path(folder.strip()).expanduser()
        if not root.is_absolute():
            raise ValueError("Type the full path of the project folder (starting with /).")
        if not root.is_dir():
            raise ValueError(f"{root} is not a folder.")
        if persistence.project_config_path(root) is None:
            raise ValueError(f"{root} has no project settings file ({root.name}.json). Choose the project's own folder.")
    root = root.resolve()
    project = _context(root)
    config_name = (persistence.load_project_config(str(root)) or {}).get("project_name") or root.name
    persistence.save_project_to_index(config_name, str(root))
    persistence.save_active_session(config_name)  # a restart opens it too
    where = {key: value for key, value in (view["spec"] or {}).items() if key in ("api", "api_port")}  # this window's server
    spec = {**where, "id": "studio-empty-" + channel, "title": "Choose an image", "layers": [],
            "session": channel, "standalone": True}
    registry.publish(channel, spec, project=project)
    registry.choose_project(channel, str(root))
    return {"name": root.name, "root": str(root)}


def create(channel, name, parent):
    """A new, empty project (the same folders and settings as classic Tab 0 makes), opened at once."""
    from spacemap_gui.tabs.tab0_project import create_project_structure

    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name.strip()):
        raise ValueError("Name the project with letters, digits, - and _ only (no spaces).")
    if not isinstance(parent, str) or not parent.strip() or not Path(parent.strip()).expanduser().is_absolute():
        raise ValueError("Type the full path of the folder to put the project in.")
    parent_path = Path(parent.strip()).expanduser()
    if not parent_path.is_dir():
        raise ValueError(f"{parent_path} is not a folder.")
    target = parent_path / name.strip() if parent_path.name == "project" else parent_path / "project" / name.strip()
    if target.exists():
        raise ValueError(f"{target} already exists. Open it instead, or choose another name.")
    root = create_project_structure(name.strip(), str(parent_path))
    return open_project(channel, folder=root)


def forget(name):
    """Take a project off the list (its folder is not touched)."""
    from spacemap_gui import persistence

    if name not in persistence.load_projects_index():
        raise ValueError("That project is not in the list.")
    persistence.remove_project_from_index(name)


def notes(root):
    """The project's description, PI and institution, and when it was made."""
    from .settings import read_config

    config = read_config(root)
    return {"description": config.get("description") or "", "pi": config.get("pi") or "",
            "institution": config.get("institution") or "", "created": (config.get("created_at") or "")[:10]}


def save_notes(root, changes):
    """Keep the description, PI or institution in the project's settings (the same keys as classic Tab 0)."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or not changes or set(changes) - set(NOTES):
        raise ValueError("Nothing to save.")
    for key, value in changes.items():
        if not isinstance(value, str) or len(value) > (4000 if key == "description" else 200):
            raise ValueError("That text is too long.")
    with project_lock(root):
        config = read_config(root)
        for key, value in changes.items():
            persistence.update_stored_config_key(config, NOTES[key], value.strip())
        persistence.save_project_config(str(root), config)
    return notes(root)
