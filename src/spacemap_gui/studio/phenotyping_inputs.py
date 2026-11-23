"""Cell tables for Phenotyping, with optional image grouping and Batch correction."""

from . import project_files


from pathlib import Path

from .settings import project_lock, read_config


def _path(root, value):
    path = Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def folders(root, mappings, table=None):
    from .cell_inputs import image_folder

    mappings = {**mappings, **read_config(root).get("directory_mappings", {})}
    return {
        "images": str(_path(root, mappings["viz_roi_dir"])) if mappings.get("viz_roi_dir") else str(image_folder(root, mappings)),
        "tables": str(_path(root, mappings.get("global_pheno_dir") or (Path(table).parent if table else mappings.get("bc_output_dir") or "outputs/batch_corrected"))),
        "output": str(_path(root, mappings.get("phenotyping_output_dir") or "outputs/phenotyping")),
    }


def table_files(folder):
    """CSV and direct Parquet inputs; a CSV cache is not a second dataset."""
    folder = Path(folder)
    return [str(path) for path in sorted(folder.iterdir(), key=lambda p: p.name, reverse=True)
            if path.is_file() and not project_files.excluded(path) and path.suffix.lower() in {".csv", ".parquet"}
            and not path.name.startswith(".")
            and not path.name.endswith(".labels.parquet")
            and not (path.name.endswith(".csv.parquet") and path.with_suffix("").is_file())] if folder.is_dir() else []


def column_roles(root, table, available=None):
    from spacemap import dataio
    from .correction import last_run

    columns = set(available if available is not None else dataio.columns(table)) if table else set()
    config = read_config(root)
    saved = table_roles(config, root, table)
    if table and saved.get("table") and _path(root, saved["table"]) == Path(table).resolve():
        roles = {key: saved.get(key, "") if saved.get(key, "") in columns else "" for key in ("roi", "slide", "cell_id")}
        return roles
    manifest = last_run(table) or {}
    bc = manifest.get("config", {})
    roles = {key: next((c for c in candidates if c in columns), "") for key, candidates in {
        "roi": [bc.get("batch_id_col"), "batch_id", "roi", "roi_id"],
        "slide": ["slide", "slide_id"],
        "cell_id": [bc.get("cell_id_col"), "cell_id"],
    }.items()}
    if not roles["roi"] and not roles["cell_id"] and "label" in columns:
        roles["cell_id"] = "label"
    return roles


def table_roles(config, root, table):
    pheno = config.get("tab_configurations", {}).get("tab7_phenotyping", {})
    choices = [pheno.get("studio_table_columns", {}), *pheno.get("studio_table_columns_by_table", {}).values()]
    return next((item for item in choices if table and item.get("table")
                 and _path(root, item["table"]) == Path(table).resolve()), {})


def single_image(root, table, images=None):
    """Match a per-image table without treating tissue-piece IDs as image names."""
    from spacemap.viewer.catalog import sample_of
    from . import cell_inputs

    if not table:
        return ""
    if images is None:
        locations = folders(root, {}, table)
        images = cell_inputs.images(root, {"bc_images_dir": locations["images"]})
    config = read_config(root)
    pairs = config.get('tab_configurations', {}).get('tab7_phenotyping', {}).get('studio_image_tables', {})
    explicit = [name for name, path in pairs.items() if path and _path(root, path) == Path(table).resolve() and name in images]
    if explicit:
        return explicit[0] if len(explicit) == 1 else ''
    saved = table_roles(config, root, table)
    if saved.get("table") and _path(root, saved["table"]) == Path(table).resolve() and saved.get("image") in images:
        return saved["image"]
    return match_table_image(table, images)


def match_table_image(table, images):
    """Conservative filename matching shared by Phenotyping and Spatial.

    Ignore separators/case and known pipeline suffixes. Fuzzy spelling matches
    must preserve every numeric identifier and have a clear winner.
    """
    import re
    from difflib import SequenceMatcher
    from spacemap.viewer.catalog import sample_of
    name = sample_of(str(table))
    if name in images:
        return name
    tokens = lambda text: re.findall(r'[a-z]+|\d+', sample_of(str(text)).lower())
    own = tokens(name)
    numbers = [v for v in own if v.isdigit()]
    candidates = [(image, tokens(image)) for image in images]
    exact = [image for image, words in candidates if words == own]
    if exact:
        return exact[0] if len(exact) == 1 else ''
    scores = sorted(((SequenceMatcher(None, ''.join(own), ''.join(words)).ratio(), image)
                     for image, words in candidates if [v for v in words if v.isdigit()] == numbers), reverse=True)
    return scores[0][1] if scores and scores[0][0] >= .9 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= .1) else ''


def image_tables(root, table=None, images=None):
    """One explicit or uniquely filename-matched table per image; never guess a tie."""
    from . import cell_inputs
    locations = folders(root, {}, table)
    if images is None:
        images = cell_inputs.images(root, {"bc_images_dir": locations["images"]})
    files = table_files(locations['tables'])
    saved = read_config(root).get('tab_configurations', {}).get('tab7_phenotyping', {}).get('studio_image_tables', {})
    candidates = {}
    for path in files:
        image = single_image(root, path, images)
        if image:
            candidates.setdefault(image, []).append(path)
    pairs = {}
    for image in images:
        if image in saved:
            path = str(_path(root, saved[image])) if saved[image] else ''
            pairs[image] = path if path in files else ''
        else:
            paths = candidates.get(image, [])
            pairs[image] = paths[0] if len(paths) == 1 else ''
    return pairs


def save_pair(root, active, image, table):
    """Pairing a per-image file also clears a mistaken tissue-piece grouping."""
    from spacemap import dataio
    from spacemap_gui import persistence
    pairs = image_tables(root, active)
    if image not in pairs:
        raise ValueError('Choose a listed image.')
    path = str(_path(root, table)) if table else ''
    if path and path not in table_files(folders(root, {}, active)['tables']):
        raise ValueError('Choose a table from the Tables folder.')
    roles = column_roles(root, path) if path else {}
    if path:
        roles['roi'] = ''
        if not roles['cell_id'] and 'label' in dataio.columns(path):
            roles['cell_id'] = 'label'
        save_roles(root, path, {**roles, 'image': image, 'label': segmentation_column(root, path)})
    with project_lock(root):
        config = read_config(root)
        saved = config.setdefault('tab_configurations', {}).setdefault('tab7_phenotyping', {}).setdefault('studio_image_tables', {})
        # One per-image table cannot belong to two images.
        for other, candidate in pairs.items():
            if path and other != image and candidate == path:
                saved[other] = ''
        saved[image] = path
        persistence.save_project_config(str(root), config)


def paired_summary(root, active, image):
    from spacemap import dataio
    from . import phenotyping_overview
    table = image_tables(root, active).get(image)
    if not table:
        raise ValueError('Pair a cell table with this image first.')
    roles = column_roles(root, table)
    if roles['roi']:
        raise ValueError('For a per-image table, choose No image column in Column roles.')
    columns = dataio.columns(table)
    info = phenotyping_overview.overview(table, roi_column='', slide_column=roles['slide'], single_roi=image)
    methods = phenotyping_overview.method_columns(root, table, columns)
    results = phenotyping_overview.results(table, suffix=None, roi_column='', single_roi=image,
        extra_columns=[c for m in methods for c in m['columns']])
    return {'roi': image, 'table_path': table, 'cells': info['cells'],
            'slide': info['rois'][0].get('slide', '') if info['rois'] else '',
            'methods': methods, 'results': {name: values.get(image, 0) for name, values in results.items()}}


def segmentation_column(root, table):
    from spacemap import dataio
    saved = table_roles(read_config(root), root, table)
    if table and saved.get("table") and _path(root, saved["table"]) == Path(table).resolve() and saved.get("label"):
        return saved["label"]
    return "label" if table and "label" in dataio.columns(table) else ""


def save_roles(root, table, roles):
    from spacemap import dataio
    from spacemap_gui import persistence

    if (not table or not isinstance(roles, dict) or not {"roi", "slide", "cell_id"} <= set(roles)
            or set(roles) - {"roi", "slide", "cell_id", "label", "image"}):
        raise ValueError("Choose the image and cell columns for this table.")
    columns = dataio.columns(table)
    if any(not isinstance(v, str) or key != "image" and v and v not in columns for key, v in roles.items()):
        raise ValueError("Choose columns that exist in this table.")
    if roles.get("image"):
        from . import cell_inputs
        images = cell_inputs.images(root, {"bc_images_dir": folders(root, {}, table)["images"]})
        if roles["image"] not in images:
            raise ValueError("Choose an image from the Images folder.")
    if not roles["cell_id"]:
        raise ValueError("Choose a unique Cell ID column. Image and Slide columns are optional.")
    identities = [roles[key] for key in ("roi", "slide", "cell_id") if roles[key]]
    if len(set(identities)) != len(identities):
        raise ValueError("Image, Slide and Cell ID must use different columns.")
    ids = dataio.load(table, columns=[roles["cell_id"]], build_fast_copy=False)[roles["cell_id"]]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError("The Cell ID column has missing or repeated IDs. Choose IDs unique across this table.")
    with project_lock(root):
        config = read_config(root)
        pheno = config.setdefault("tab_configurations", {}).setdefault("tab7_phenotyping", {})
        history = pheno.setdefault("studio_table_columns_by_table", {})
        previous = pheno.get("studio_table_columns", {})
        if previous.get("table"):
            history[str(_path(root, previous["table"]))] = previous
        saved = {"table": str(Path(table).resolve()), **roles}
        pheno["studio_table_columns"] = saved
        history[saved["table"]] = saved
        persistence.save_project_config(str(root), config)


def use_table(channel, path):
    """Persist an explicitly selected input and immediately use it in this Studio session."""
    from spacemap import dataio
    from spacemap_gui import persistence
    from spacemap_gui.viewer import registry

    project = registry.session_project(channel) or {}
    root = project.get("root")
    if not root or project.get("read_only"):
        raise ValueError("Open an editable project before choosing its cell table.")
    path = _path(root, path)
    if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Choose an existing CSV or Parquet cell table.")
    dataio.columns(str(path))  # fail before saving an unreadable file
    with project_lock(root):
        config = read_config(root)
        for key, value in {"global_pheno_dir": str(path.parent), "global_pheno_file": path.name,
                           "global_pheno_dir_user_override": True, "proj_unified_csv_filename": path.name}.items():
            persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
    registry.session_use_table(channel, str(path))
    return str(path)


def save_folders(root, changes, *, output_key="phenotyping_output_dir"):
    from spacemap_gui import persistence

    keys = {"images": "viz_roi_dir", "tables": "global_pheno_dir", "output": output_key}
    if not isinstance(changes, dict) or set(changes) - set(keys):
        raise ValueError("Choose an Images, Tables or Output folder.")
    checked = {}
    for name, value in changes.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError("Enter a folder path.")
        path = _path(root, value.strip())
        if path.exists() and not path.is_dir():
            raise ValueError(f"{path.name} is a file. Enter its folder here.")
        if name != "output" and not path.is_dir():
            raise ValueError(f"The {name} folder does not exist.")
        checked[keys[name]] = str(path)
    with project_lock(root):
        config = read_config(root)
        for key, value in checked.items():
            persistence.update_stored_config_key(config, key, value)
        if "tables" in changes:
            persistence.update_stored_config_key(config, "global_pheno_dir_user_override", True)
        persistence.save_project_config(str(root), config)


def plan(root, mappings, table=None, *, summaries=True):
    from spacemap import dataio
    from spacemap.viewer.catalog import sample_of
    from . import cell_inputs, phenotyping_overview

    locations = folders(root, mappings, table)
    warnings = []
    try:
        images = cell_inputs.images(root, {"bc_images_dir": locations["images"]})
    except (OSError, ValueError) as error:
        images = {}
        warnings.append(str(error))
    available, roles = [], {"roi": "", "slide": "", "cell_id": ""}
    info, results, methods = {"table": "", "cells": 0, "rois": [], "phenotypes": []}, {}, []
    image = ""
    try:
        available = dataio.columns(table) if table and Path(table).is_file() else []
        roles = column_roles(root, table, available)
        image = single_image(root, table, images) if not roles["roi"] else ""
        # Controls and image scope need schema/pairings, not millions of labels
        # or archived job profiles. Summaries arrive independently in the UI.
        if summaries or not images:
            info = phenotyping_overview.overview(table, roi_column=roles["roi"], slide_column=roles["slide"], single_roi=image)
        else:
            info = {"table": Path(table).name if table else "", "cells": None, "rois": [],
                    "phenotypes": [c for c in available if c.startswith('phenotype_') or c in {'consensus', 'cluster', 'leiden'}]}
        if summaries:
            methods = phenotyping_overview.method_columns(root, table, available)
            results = phenotyping_overview.results(table, suffix=None, roi_column=roles["roi"],
                                                   extra_columns=[c for m in methods for c in m["columns"]], single_roi=image)
    except (OSError, ValueError, KeyError, dataio.DataIOError) as error:
        warnings.append(str(error))
    counts = {sample_of(item["roi"]): item["cells"] for item in info["rois"]}
    slides = {sample_of(item["roi"]): item.get("slide", "") for item in info["rois"]}
    coverage = {}
    for column, seen in results.items():
        grouped = coverage[column] = {}
        for name, n in seen.items():
            roi = sample_of(name)
            grouped[roi] = grouped.get(roi, 0) + n
    pairs = image_tables(root, table, images) if not roles['roi'] else {}
    per_image = not roles['roi'] or bool(image and any(pairs.values()))
    if available and images:
        if summaries and roles["roi"] and not set(counts) & set(images):
            warnings.append("Image names do not match. For one image per table, choose ‘No image column’.")
        elif not roles["roi"] and not image:
            warnings.append("Choose the image for this table under Column roles.")
    rows = [{"roi": roi, "image": Path(images[roi]).name if roi in images else "",
             "image_path": images.get(roi, ""), "cells": counts.get(roi), "slide": slides.get(roi, ""),
             "table_path": pairs.get(roi, '') if per_image else str(table or ''),
             "results": {column: seen.get(roi, 0) for column, seen in coverage.items()}}
            for roi in sorted(images if images else counts)]
    if per_image:
        for row in rows:
            if row['table_path'] != str(table or ''):
                row.update(cells=None, slide='', results={})
    from . import phenotyping_images
    return {**info, "summaries_ready": summaries, "selected_images": phenotyping_images.selected(root, table, rows, per_image) if table else [],
            "path": str(table or ""), "folders": locations,
            "single_image": image, "per_image": per_image,
            "segmentation_label": segmentation_column(root, table) if available else "",
            "tables": table_files(locations["tables"]), "rows": rows,
            "cell_id_column": roles["cell_id"], "columns": available, "roles": roles,
            "result_columns": list(results), "methods": methods, "warnings": warnings}


def combined_table(channel):
    from spacemap_gui.viewer import registry

    project = registry.session_project(channel) or {}
    table = project.get("unified")
    if not table or not Path(table).is_file():
        raise ValueError("Choose a cell table at the top of Phenotyping.")
    return str(table)
