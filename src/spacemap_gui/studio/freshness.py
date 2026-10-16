"""File-age hints for tagging, and provenance-based change checks for combined results.

An old combined table without recorded inputs is unverified, not presumed to
belong to the currently selected segmentation dataset.
"""

from pathlib import Path


def _stamp(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def check(root, config, mappings, combined=None):
    """Tagging age hints, changed recorded combined inputs, and an unverified flag for legacy results."""
    from spacemap_gui.tabs import cell_tagging_ui as tab5

    from . import segmentation, tagging

    places = segmentation.step_folders(root, config, mappings)
    output = places["output"]
    state = tagging.values(root, config)
    tagged_folder, suffix = Path(state[tab5.OUTPUT_DIR_KEY]), state[tab5.OUTPUT_SUFFIX_KEY]
    combined_stamp = _stamp(combined) if combined else None
    stale_tags, stale_combined = [], []
    tables = sorted(output.glob("*_cell_features.csv")) if output.is_dir() else []
    for table in tables:
        sample = tab5.sample_id_from_csv(table.name, "")
        made = _stamp(table)
        tagged = tagged_folder / f"{sample}{suffix}.csv"
        from spacemap.viewer.catalog import sample_of
        sample = sample_of(table.name)
        tagged_at = _stamp(tagged)
        if tagged_at is not None and made is not None and tagged_at < made:
            stale_tags.append(sample)
    # Age alone does not prove that a legacy table was derived from this dataset.
    from .correction import last_run, _stamp as input_stamp
    manifest = last_run(combined)
    inputs = (manifest or {}).get("inputs") or {}
    for roi, recorded in inputs.items():
        try:
            unchanged = input_stamp(recorded["path"]) == recorded
        except (KeyError, OSError):
            unchanged = False
        if not unchanged:
            stale_combined.append(roi)
    return {"tagging": stale_tags, "combined": stale_combined, "table": Path(combined).name if combined else "",
            "combined_path": str(combined or ""), "unverified": combined_stamp is not None and not inputs,
            "segmentation": str(output)}
