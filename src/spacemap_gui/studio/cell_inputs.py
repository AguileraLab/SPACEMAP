"""Match ROI images to measurement tables without turning table suffixes into samples."""

from pathlib import Path

from spacemap.viewer.catalog import sample_of


def image_folder(root, mappings):
    from . import rois
    from .settings import read_config
    from .lineage import _path

    value = (mappings or {}).get("bc_images_dir")
    return _path(root, value) if value else rois.folders(root, mappings, read_config(root))[2]


def images(root, mappings):
    from .rois import IMAGE_SUFFIXES

    folder = image_folder(root, mappings)
    found = {}
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        if path.is_file() and not path.name.startswith(".") and path.name.lower().endswith(IMAGE_SUFFIXES):
            roi = sample_of(path.name)
            if roi in found:
                raise ValueError(f"Two ROI images have the same name {roi}. Choose a folder with one image per ROI.")
            found[roi] = str(path)
    return found


def table_index(folder):
    """ROI -> candidates. CSV and its Parquet sibling are one logical table."""
    folder = Path(folder)
    logical = {}
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in (".csv", ".parquet"):
            continue
        if path.name.lower().endswith(".csv.parquet"):
            if path.with_suffix("").is_file():
                continue  # a verified cache is used at read time
            stem = path.with_suffix("").stem
        else:
            stem = path.stem
        logical.setdefault(stem, []).append(path)
    found = {}
    for paths in logical.values():
        # Prefer Parquet when it is at least as recent as the corresponding CSV.
        chosen = max(paths, key=lambda p: (p.stat().st_mtime_ns, p.suffix.lower() == ".parquet"))
        found.setdefault(sample_of(chosen.name), []).append(str(chosen))
    return found


def read_path(path):
    """Use an existing verified Parquet cache, without creating one as a side effect."""
    from spacemap import dataio

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return path
    info = dataio.fast_copy_info(path)
    return Path(info["twin"]) if info.get("current") else path


def read_frame(path, columns=None, text_columns=()):
    import pandas as pd

    source = read_path(path)
    frame = pd.read_parquet(source, columns=columns) if source.suffix.lower() == ".parquet" else pd.read_csv(source, usecols=columns, converters={name: str for name in text_columns})
    if source != Path(path) and text_columns:
        # CSV caches can infer numeric types or nulls for identifier strings such as 001 or NA.
        # Read only those keys from the CSV to preserve their exact identity.
        keys = pd.read_csv(path, usecols=list(text_columns), dtype=str, keep_default_na=False)
        for column in text_columns:
            frame[column] = keys[column]
    return frame


def matches(root, mappings, folder):
    sources = table_index(folder)
    rows = []
    for roi, image in images(root, mappings).items():
        candidates = sources.get(roi, [])
        rows.append({"roi": roi, "image": Path(image).name, "image_path": image,
                     "path": candidates[0] if len(candidates) == 1 else "",
                     "candidates": candidates,
                     "match": "matched" if len(candidates) == 1 else "ambiguous" if candidates else "missing"})
    return rows
