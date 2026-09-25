"""Which folder a step reads from, following what was run before it.

Tab 3 (ROI Extraction) reads the images of the last step that produced any:
background-subtracted images from Tab 2 when that step was run, otherwise the
registered images from Tab 1. A folder you choose yourself is kept, unless it
is Tab 3's own output folder (the ROIs it writes), which is never a valid input.

Tab 3's input used to be tied to Tab 2's output by a folder link, so choosing
one changed the other: pointing Tab 3 at the ROI folder made Tab 2 write there.
``unlink_step_outputs`` repairs projects saved that way.
"""

from __future__ import annotations

import os
from typing import Mapping

IMAGE_SUFFIXES = (".tif", ".tiff", ".qptiff", ".svs")
BACKGROUND_DEFAULT = "${PROJECT_ROOT}/outputs/background_subtracted"


def same_folder(first, second) -> bool:
    """True when two folder settings name the same folder (``/work`` and ``/endosome/work`` included)."""
    if not (isinstance(first, str) and isinstance(second, str) and first.strip() and second.strip()):
        return False
    try:
        return os.path.realpath(os.path.expanduser(first.strip())) == os.path.realpath(os.path.expanduser(second.strip()))
    except (OSError, ValueError):
        return first.strip() == second.strip()


def has_images(folder) -> bool:
    """True when *folder* directly holds at least one image file."""
    if not folder:
        return False
    try:
        with os.scandir(folder) as entries:
            return any(entry.name.lower().endswith(IMAGE_SUFFIXES) and entry.is_file() for entry in entries)
    except OSError:
        return False


def _first(mappings: Mapping, *keys: str) -> str:
    for key in keys:
        value = mappings.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def tab3_input_suggestion(mappings: Mapping) -> tuple[str, str]:
    """(folder, why) Tab 3 should read from; ("", "") when no earlier step has a folder set."""
    rois = _first(mappings, "tab3_output_dir", "tab3_multi_output_dir", "roi_output_dir")
    background = _first(mappings, "bg_subtraction_output_dir", "tab2_output_dir")
    registered = _first(mappings, "tab1_batch_output_dir", "registration_output_dir", "tab2_input_dir_reg")
    if background and not same_folder(background, rois) and has_images(background):
        return background, "the background-subtracted images from Tab 2"
    if registered:
        return registered, "the registered images from Tab 1 (no background-subtracted images yet)"
    return "", ""


def follow_pipeline(current: str | None, mappings: Mapping) -> tuple[str | None, str]:
    """Tab 3's input folder: (value to use, note to show when it was changed)."""
    suggestion, why = tab3_input_suggestion(mappings)
    if not suggestion:
        return current, ""
    rois = _first(mappings, "tab3_output_dir", "tab3_multi_output_dir", "roi_output_dir")
    earlier_steps = [_first(mappings, "bg_subtraction_output_dir", "tab2_output_dir"),
                     _first(mappings, "tab1_batch_output_dir", "registration_output_dir", "tab2_input_dir_reg")]
    following = (not current or not str(current).strip() or same_folder(current, rois)
                 or any(same_folder(current, folder) for folder in earlier_steps if folder))
    if following and not same_folder(current, suggestion):
        return suggestion, f"Images Directory follows the pipeline: {why}."
    return current, ""


def unlink_step_outputs(config: dict, project_root: str) -> str:
    """Set Tab 2's output back when it points at Tab 3's ROI folder. Returns the new value ("" if unchanged)."""
    def expand(value: str) -> str:
        return value.replace("${PROJECT_ROOT}", str(project_root))

    mappings = config.get("directory_mappings") if isinstance(config.get("directory_mappings"), dict) else {}
    portable = config.get("portable_state") if isinstance(config.get("portable_state"), dict) else {}
    interface = portable.get("interface_state") if isinstance(portable.get("interface_state"), dict) else {}
    rois = _first(mappings, "tab3_output_dir", "tab3_multi_output_dir", "roi_output_dir") or \
        _first(interface, "tab3_output_dir", "tab3_multi_output_dir", "roi_output_dir")
    if not rois:
        return ""
    fixed = ""
    for section in (mappings, interface):
        for key in ("tab2_output_dir", "bg_subtraction_output_dir"):
            value = section.get(key)
            if isinstance(value, str) and value.strip() and same_folder(expand(value), expand(rois)):
                replacement = BACKGROUND_DEFAULT if "${PROJECT_ROOT}" in value else expand(BACKGROUND_DEFAULT)
                section[key] = replacement
                fixed = replacement
    return fixed
