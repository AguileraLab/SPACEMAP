"""Cells in Layers: the Segmentation step's latest outlines (even after its folder changed), and imported ones."""

import json

from test_viewer import _square, _write_channel_pages


def _outlines(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection",
                                "features": [_square(i + 1, 20 * i + 5, 10) for i in range(count)]}))


def _project(tmp_path):
    from spacemap_gui.studio.settings import config_path

    root = tmp_path / "study"
    image = root / "outputs/rois/P11_roi_1.ome.tiff"
    image.parent.mkdir(parents=True)
    _write_channel_pages(image, channels=2, height=200, width=200)
    _outlines(root / "outputs/segmentation/geojson/P11_roi_1.ome_cell_boundaries.geojson", 3)       # the old run
    _outlines(root / "outputs/segmentation_new/geojson/P11_roi_1.ome_cell_boundaries.geojson", 5)   # the new run
    config = {"directory_mappings": {"tab4_input_dir": "${PROJECT_ROOT}/outputs/rois",
                                     "tab4_output_dir": "${PROJECT_ROOT}/outputs/segmentation_new"}}
    config_path(root).write_text(json.dumps(config))
    entry = {"group": "ROIs", "name": image.name, "path": str(image), "sample": "P11_roi_1"}
    return root, image, entry


def test_the_image_shows_the_segmentation_steps_latest_cells_after_its_folder_changed(tmp_path):
    from spacemap_gui.viewer import registry, ui

    root, image, entry = _project(tmp_path)
    # The session was opened before Segmentation's output changed: it still says outputs/segmentation.
    stale = {"root": str(root), "mappings": {"tab4_output_dir": "${PROJECT_ROOT}/outputs/segmentation"}}
    with ui.using_context(stale):
        layers, _ = ui.layers_for_image(entry)
    cells = next(layer for layer in layers if layer["type"] == "cells")
    assert "segmentation_new" in registry.resolve_object(cells["token"], "cells")["segmentation"]


def test_imported_cells_are_their_own_removable_layer(tmp_path):
    from spacemap_gui.studio import cell_layers
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer import registry, ui

    root, image, entry = _project(tmp_path)
    config = read_config(root)
    mappings = {k: v.replace("${PROJECT_ROOT}", str(root)) for k, v in config["directory_mappings"].items()}
    found = cell_layers.candidates(root, config, mappings, str(image), "P11_roi_1")
    names = [item["name"] for item in found]
    assert "outputs/segmentation/geojson/P11_roi_1.ome_cell_boundaries.geojson" in names
    old = next(item["path"] for item in found if "segmentation/geojson" in item["name"])
    cell_layers.add(root, str(image), old)
    with ui.using_context({"root": str(root), "mappings": {}}):
        layers, _ = ui.layers_for_image(entry)
    extra = [layer for layer in layers if layer.get("imported_cells")]
    assert len(extra) == 1 and extra[0]["added"] and extra[0]["id"].startswith("cells-")
    assert registry.resolve_object(extra[0]["token"], "cells")["segmentation"] == old
    cell_layers.remove(root, str(image), old)
    assert cell_layers.imported(root, str(image)) == []


def test_the_outlines_folder_is_geojson_in_output_unless_changed(tmp_path):
    from spacemap_gui.studio import segmentation

    root = tmp_path / "study"
    root.mkdir()
    places = segmentation.step_folders(root, {}, {})
    assert places["geojson"] == places["output"] / "geojson"
    config = {"tab_configurations": {"tab5_segmentation": {segmentation.GEOJSON_KEY: "${PROJECT_ROOT}/outlines"}}}
    places = segmentation.step_folders(root, config, {})
    assert places["geojson"] == root / "outlines"
    table, outlines = segmentation.results(places["output"], "a.ome.tiff", places["geojson"])
    assert outlines == root / "outlines" / "a.ome_cell_boundaries.geojson" and table.parent == places["output"]


def test_new_cells_colour_from_their_own_table_not_an_older_combined_one(tmp_path):
    """A combined table made before a new segmentation numbers other cells: the layer offers the new run's own
    table first (with batch_id added for the viewer) and names the old one."""
    import os
    import time

    import pandas as pd

    from spacemap import dataio
    from spacemap_gui.viewer import registry, ui

    root, image, entry = _project(tmp_path)
    combined = root / "outputs/batch_corrected/combined.csv"
    combined.parent.mkdir(parents=True)
    pd.DataFrame({"batch_id": ["P11_roi_1"] * 5, "label": range(1, 6), "epithelia_bed": [1, 0, 1, 0, 1],
                  "x_centroid": [400, 300, 200, 100, 0], "y_centroid": range(5)}).to_csv(combined, index=False)  # other cells
    old = time.time() - 3600
    os.utime(combined, (old, old))
    table = root / "outputs/segmentation_new/P11_roi_1.ome_cell_features.csv"
    pd.DataFrame({"label": range(1, 6), "area": [10.0] * 5, "x_centroid": range(5), "y_centroid": range(5)}).to_csv(table, index=False)
    with ui.using_context({"root": str(root), "mappings": {}, "unified": str(combined)}):
        layers, _ = ui.layers_for_image(entry)
    cells = next(layer for layer in layers if layer["type"] == "cells")
    names = [item["name"] for item in cells["tables"]]
    assert names[0] == "This segmentation's cells" and cells["table"] == cells["tables"][0]["id"] and not cells["color_by"]
    assert len(names) == 1 and "another segmentation" in cells["hidden_tables"][0]["reason"]  # not offered, and said why
    own = registry.resolve_object(cells["token"], "cells")["tables"][cells["tables"][0]["id"]]
    assert "batch_id" in dataio.columns(own)


def test_cell_data_as_circles_voronoi_and_the_images_cells(tmp_path):
    import pandas as pd
    from shapely.geometry import box, mapping, shape

    from spacemap_gui.studio import annotation, annotation_edits as edits, cell_data, cell_layers
    from spacemap_gui.studio.settings import read_config
    from spacemap_gui.viewer import ui

    root, image, entry = _project(tmp_path)
    config = read_config(root)
    mappings = {k: v.replace("${PROJECT_ROOT}", str(root)) for k, v in config["directory_mappings"].items()}
    # Squares 1..5 of the new run sit at x = 20 i + 5 (20 px wide), y = 10..30: one table row inside each, two images.
    table = root / "data/cells.csv"
    table.parent.mkdir(parents=True)
    rows = [{"image": "P11_roi_1", "label": i + 1, "X µm": (20 * i + 15) * 0.5, "Y µm": 20 * 0.5, "score": float(i)} for i in range(5)]
    rows += [{"image": "P2_roi_9", "label": 1, "X µm": 5.0, "Y µm": 5.0, "score": 9.0}]
    pd.DataFrame(rows).to_csv(table, index=False)
    facts = cell_data.inspect(table, "P11_roi_1")
    assert (facts["x"], facts["y"], facts["units"], facts["roi_column"], facts["roi_value"], facts["label_column"]) == (
        "X µm", "Y µm", "um", "image", "P11_roi_1", "label")
    base = {"x": "X µm", "y": "Y µm", "units": "um", "roi_column": "image", "roi_value": "P11_roi_1", "label_column": "label", "within": ""}
    own = cell_layers.own(root, config, mappings, str(image), "P11_roi_1")
    cache = ui._cache_dir(root)
    circles = cell_data.build(root, config, mappings, str(image), table, {**base, "show": "circles", "size_um": 4}, cache, own)
    drawn = json.loads(open(circles["geojson"]).read())["features"]
    assert circles["kept"] == 5 and len(drawn) == 5 and abs(shape(drawn[0]["geometry"]).area - 3.06 * 16) < 5  # 4 µm = 8 px across
    voronoi = cell_data.build(root, config, mappings, str(image), table, {**base, "show": "voronoi", "reach_um": 2}, cache, own)
    assert len(json.loads(open(voronoi["geojson"]).read())["features"]) == 5
    by_number = cell_data.build(root, config, mappings, str(image), table, {**base, "show": "cells", "link": "label"}, cache, own)
    by_place = cell_data.build(root, config, mappings, str(image), table, {**base, "show": "cells", "link": "position"}, cache, own)
    assert by_number["matched"] == 5 and by_place["matched"] == 5
    linked = pd.read_parquet(by_place["table"])
    assert sorted(linked["label"]) == [1, 2, 3, 4, 5] and set(linked["batch_id"]) == {"P11_roi_1"}
    # Only inside an annotation: a tissue over the first two squares.
    output = annotation.folders(root, mappings)[1]
    tissue = annotation.valid_tissue(output, "P11_roi_1")
    edits.replace(tissue, [{"type": "Feature", "geometry": mapping(box(0, 0, 45, 60)), "properties": {"name": "tissue"}}],
                  edits.revision(tissue), "Fixture", kind="tissue")
    key = next(k for group, items in cell_data.area_choices(root, config, str(image), output) for k, name in items if name == "tissue")
    inside = cell_data.build(root, config, mappings, str(image), table, {**base, "show": "circles", "size_um": 4, "within": key}, cache, own)
    assert inside["kept"] == 2 and "inside tissue" in cell_data.describe(inside)
    # The image opens with them, and a layer can be taken off again.
    with ui.using_context({"root": str(root), "mappings": {}}):
        layers, _ = ui.layers_for_image(entry)
    assert sum(1 for layer in layers if layer.get("data_layer")) == 3  # circles, Voronoi, circles inside tissue
    cells = next(layer for layer in layers if layer["id"] == "cells")
    assert any(item["name"] == "Data: cells.csv" for item in cells["tables"])  # the linked ones join the image's cells
    cell_data.remove(root, str(image), inside["id"])
    assert not any(item["id"] == inside["id"] for item in cell_data.records(root, str(image)))
