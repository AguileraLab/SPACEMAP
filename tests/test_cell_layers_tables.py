"""An imported segmentation's cell table: the one dropped with it, else the table of the same name beside it."""
import json

from spacemap_gui.studio import cell_layers


def _outlines(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"label": 1}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}))
    return path


def test_dropped_outlines_take_the_table_dropped_with_them(tmp_path):
    image = tmp_path / "P11_roi_1.ome.tiff"
    image.write_bytes(b"x")
    dropped = tmp_path / "outputs/studio/dropped"
    outlines = _outlines(dropped / "1a2b3c4d_P11_roi_1.ome_cell_boundaries.geojson")
    table = dropped / "9f8e7d6c_P11_roi_1.ome_cell_features.csv"
    table.write_text("label,x\n1,0\n")
    assert cell_layers._cells_name(outlines) == cell_layers._cells_name(table) == "p11_roi_1.ome"
    assert cell_layers.table_of(tmp_path, image, outlines) == str(table)  # found beside it by name
    other = tmp_path / "elsewhere/cells.csv"
    other.parent.mkdir()
    other.write_text("label\n1\n")
    cell_layers.add(tmp_path, image, outlines, other)
    assert cell_layers.table_of(tmp_path, image, outlines) == str(other.resolve())  # the one given with it wins


def test_a_run_folder_pairs_outlines_with_its_table(tmp_path):
    image = tmp_path / "P11_roi_1.ome.tiff"
    image.write_bytes(b"x")
    run = tmp_path / "outputs/segmentation_new"
    outlines = _outlines(run / "geojson/P11_roi_1.ome_cell_boundaries.geojson")
    (run / "P11_roi_1.ome_cell_features.csv").write_text("label\n1\n")
    (run / "P11_roi_10.ome_cell_features.csv").write_text("label\n1\n")
    assert cell_layers.table_of(tmp_path, image, outlines) == str(run / "P11_roi_1.ome_cell_features.csv")
