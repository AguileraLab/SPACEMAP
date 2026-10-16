"""User-wide phenotype colours, shared with Tab 9 and the viewer.

Tab 9 keeps one colour per phenotype in the project as
``tab9_actual_cell_color_val_<name>``, the name with every run of characters
other than letters, digits and ``_`` turned into ``_``; the viewer colours cells
from the same keys. Studio's shared preset takes precedence over older project
choices; deliberate colour edits are persisted globally and mirrored to these
classic keys for the current project.
"""

import re

PREFIX = "tab9_actual_cell_color_val_"
HEX = re.compile(r"#[0-9a-fA-F]{6}")


def key(label):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(label)).strip("_") or "item"


def _section(config):
    return (config.get("tab_configurations") or {}).get("tab9_visualization") or {}


def saved(config):
    """Shared names win over legacy project colours, including old aliases."""
    from spacemap.phenotype_presets import canonical, colours
    shared = {key(name).casefold():value for name,value in colours().items()}
    legacy = {
        name[len(PREFIX) :]: value
        for name, value in _section(config).items()
        if name.startswith(PREFIX) and isinstance(value, str) and HEX.fullmatch(value)
    }
    return {**{name:shared.get(key(canonical(name)).casefold(),value) for name,value in legacy.items()},
            **{key(name):value for name,value in colours().items()}}


def save(root, colours):
    """Save deliberate edits globally and mirror them to classic's project keys."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(colours, dict) or not 0 < len(colours) <= 500:
        raise ValueError("Choose a colour for at least one name.")
    checked = {}
    for label, value in colours.items():
        if not isinstance(label, str) or not 0 < len(label.strip()) <= 128:
            raise ValueError("Each colour needs a name.")
        if not isinstance(value, str) or not HEX.fullmatch(value):
            raise ValueError(f"{label}: choose a colour as #rrggbb.")
        checked[PREFIX + key(label)] = value.lower()
    with project_lock(root):
        config = read_config(root)
        from spacemap import phenotype_presets
        phenotype_presets.update(colours)
        for name, value in checked.items():
            persistence.update_stored_config_key(config, name, value)
        persistence.save_project_config(str(root), config)
        return saved(read_config(root))
