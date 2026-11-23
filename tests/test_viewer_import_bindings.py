"""Adding an old table must not replace active table IDs or link another segmentation."""
from pathlib import Path

import pandas as pd
from starlette.testclient import TestClient

from test_studio_cell_layers import _project


def test_import_keeps_existing_bindings_and_exact_outlines(tmp_path, monkeypatch):
    from spacemap_gui.viewer import api, registry, ui
    from spacemap_gui.studio import cell_data, cell_layers

    root, image, entry = _project(tmp_path)
    table = root / "old_labels.csv"
    pd.DataFrame({"label": [1, 2], "x_centroid": [15, 35], "y_centroid": [20, 20],
                  "phenotype_ref1": ["Old A", "Old B"]}).to_csv(table, index=False)
    before = table.read_bytes()
    monkeypatch.setattr(api, "_studio_project", lambda *a, **k: root)
    monkeypatch.setattr(api, "_open_entry", lambda *a, **k: (entry, "fixture"))
    with ui.using_context({"root": str(root), "mappings": {}}):
        layers, _ = ui.layers_for_image(entry)
    main = next(layer for layer in layers if layer["id"] == "cells")
    old = str(root / "outputs/segmentation/geojson/P11_roi_1.ome_cell_boundaries.geojson")
    cell_layers.add(root, str(image), old)
    with ui.using_context({"root": str(root), "mappings": {}}):
        layers, _ = ui.layers_for_image(entry)
    imported = next(layer for layer in layers if layer.get("imported_cells"))
    token = imported["token"]
    obj = registry.resolve_object(token, "cells")
    original_tables = dict(obj["tables"])
    source = registry.register_object("datafile", path=str(table))
    request = {"action": "build", "image_token": main["image_token"], "file_token": source,
               "target_token": token, "target_id": imported["id"],
               "choice": {"x": "x_centroid", "y": "y_centroid", "units": "px",
                          "label_column": "label", "show": "cells", "link": "label"}}
    with TestClient(api.app) as client:
        response = client.post("/studio/fixture/cell_data", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    layer = result["layer"]
    assert layer["token"] == token and layer["id"] == imported["id"]
    assert layer["image_token"] == main["image_token"]
    assert {k: obj["tables"][k] for k in original_tables} == original_tables
    assert obj["tables"][layer["table"]] == result["record"]["table"]
    assert Path(result["record"]["segmentation"]) == Path(old)
    assert table.read_bytes() == before
    with ui.using_context({"root": str(root), "mappings": {}}):
        reopened, _ = ui.layers_for_image(entry)
    own = next(l for l in reopened if l["id"] == "cells")
    other = next(l for l in reopened if l["id"] == imported["id"])
    assert layer["table"] not in {t["id"] for t in own["tables"]}
    assert layer["table"] in {t["id"] for t in other["tables"]}
    assert registry.resolve_object(other["token"], "cells")["tables"][layer["table"]] == result["record"]["table"]
    # An otherwise valid token for a different image cannot be linked by accident.
    foreign = registry.register_object("cells", image=str(tmp_path / "other.tiff"), segmentation=old, roi="other", tables={})
    count = len(cell_data.records(root, str(image)))
    with TestClient(api.app) as client:
        response = client.post("/studio/fixture/cell_data", json={**request, "target_token": foreign})
    assert response.status_code == 400
    assert len(cell_data.records(root, str(image))) == count
