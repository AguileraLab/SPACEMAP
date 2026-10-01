"""Render every SPACEMAP page (and each Tab 7 method) on a synthetic project.

Run by tests/test_pages_render.py in a process of its own:
    python tests/page_smoke.py <repo root> <output json>
HOME and SPACEMAP_HOME point at a scratch folder, so no real project, project
list or setting is read or written. Writes {page: {exceptions, errors, ...}}.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
scratch = Path(tempfile.mkdtemp(prefix="smoke-"))
os.environ["HOME"] = str(scratch / "home")
os.environ["SPACEMAP_HOME"] = str(scratch / "spacemap_home")
os.environ["SPACEMAP_ASSISTANT"] = "off"
(scratch / "home").mkdir()
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

root = scratch / "project" / "Smoke"
for folder in ("data/codex", "data/he", "data/boundaries", "outputs/registered", "outputs/rois",
               "outputs/segmentation/geojson", "outputs/annotations", "outputs/batch_corrected",
               "outputs/spatial_analysis/neighborhoods", "outputs/phenotyping"):
    (root / folder).mkdir(parents=True)
markers = ["DAPI", "CD3e", "CD4", "CD8", "CD20", "CD45", "PanCK"]


def ome_image(path, size=600):
    ome = ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
           f'<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="{size}" SizeY="{size}" SizeC="{len(markers)}" '
           'SizeZ="1" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
           + "".join(f'<Channel ID="Channel:0:{i}" Name="{n}" SamplesPerPixel="1"/>' for i, n in enumerate(markers))
           + "".join(f'<TiffData FirstC="{i}" IFD="{i}" PlaneCount="1"/>' for i in range(len(markers)))
           + "</Pixels></Image></OME>")
    rng = np.random.default_rng(0)
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for c in range(len(markers)):
            data = (rng.random((size, size)) * 2000).astype(np.uint16)
            writer.write(data, subifds=1, tile=(256, 256), description=ome if c == 0 else None, metadata=None)
            writer.write(data[::2, ::2], subfiletype=1, tile=(256, 256))


ome_image(root / "outputs/registered/P1_merged.ome.tiff")
# Tab 1 pairs: one pair by the naming rule, and a CODEX image whose name does not follow it.
for image in ("data/codex/P1_codex.ome.tiff", "data/codex/slide7.ome.tiff", "data/he/P1_he.ome.tiff"):
    ome_image(root / image, size=300)
ome_image(root / "outputs/rois/s1r1.ome.tiff")
features, rows = [], []
rng = np.random.default_rng(1)
for n in range(1, 201):
    cx, cy = 20 + (n % 20) * 28, 20 + (n // 20) * 28
    ring = [[cx - 10, cy - 10], [cx + 10, cy - 10], [cx + 10, cy + 10], [cx - 10, cy + 10], [cx - 10, cy - 10]]
    features.append({"type": "Feature", "properties": {"label": n}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    rows.append({"cell_id": f"s1r1_{n}", "batch_id": "s1r1", "label": n, "x_centroid": cx, "y_centroid": cy,
                 **{m: float(rng.gamma(2, 50)) for m in markers},
                 "phenotype_leiden": ["T", "B", "Tumour", "Macrophage"][n % 4], "neighborhood": f"CN{n % 3}"})
(root / "outputs/segmentation/geojson/s1r1.ome_cell_boundaries.geojson").write_text(
    json.dumps({"type": "FeatureCollection", "features": features}))
(root / "outputs/annotations/s1r1_tissue_classes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"classification": {"name": "tumour", "color": [255, 0, 0]}},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [300, 0], [300, 300], [0, 300], [0, 0]]]}}]}))
table = pd.DataFrame(rows)
table.to_csv(root / "outputs/batch_corrected/unified.csv", index=False)
table[["cell_id", "batch_id", "label", "x_centroid", "y_centroid", "phenotype_leiden", "neighborhood"]].to_csv(
    root / "outputs/spatial_analysis/neighborhoods/neighborhood_cell_level.csv", index=False)
config = {"project_name": "Smoke", "project_root": str(root), "unified_csv_filename": "unified.csv",
          "codex_raw_dir": str(root / "data/codex"), "he_raw_dir": str(root / "data/he"),
          "directory_mappings": {"bc_output_dir": str(root / "outputs/batch_corrected"),
                                 "batch_codex_dir": str(root / "data/codex"), "batch_he_dir": str(root / "data/he"),
                                 "tab1_batch_output_dir": str(root / "outputs/registered"),
                                 "tab3_output_dir": str(root / "outputs/rois"),
                                 "tab4_geojson_dir": str(root / "outputs/segmentation/geojson")},
          "tab_configurations": {}, "audit_log": []}
(root / "Smoke.json").write_text(json.dumps(config))

gui = REPO / "src/spacemap_gui"
pages = [("tab0", "tabs/tab0_project.py"), ("tab1", "tabs/tab1_registration.py"),
         ("tab2", "tabs/tab2_background_subtraction.py"), ("tab3", "tabs/tab3_roi_extraction.py"),
         ("tab4", "tabs/tab4_annotation/__init__.py"), ("tab5", "tabs/tab5_segmentation.py"),
         ("tab6", "tabs/tab6_batch_correction.py"), ("tab8", "tabs/tab8_analysis/__init__.py"),
         ("tab9", "tabs/tab9_visualization/__init__.py")]
pages += [(f"tab7:{sub}", "tabs/tab7_phenotyping/__init__.py") for sub in
          ("leiden", "som", "vortex", "resolve", "scimap", "tools", "mrml", "consensus_ml", "subphenotyping")]
pages.append(("app", "app.py"))
results = {}
for name, relative in pages:
    started = time.perf_counter()
    try:
        at = AppTest.from_file(str(gui / relative), default_timeout=900)
        if name != "app":
            at.session_state["active_project"] = json.loads(json.dumps(config))
            at.session_state["active_project_root"] = str(root)
            at.session_state["active_project_name"] = "Smoke"
            at.session_state["global_pheno_dir"] = str(root / "outputs/batch_corrected")
            at.session_state["global_pheno_file"] = "unified.csv"
            at.session_state["global_spatial_dir"] = str(root / "outputs/batch_corrected")
            at.session_state["global_spatial_file"] = "unified.csv"
            at.session_state["global_visualization_dir"] = str(root / "outputs/batch_corrected")
            at.session_state["global_visualization_file"] = "unified.csv"
        if name.startswith("tab7:"):
            at.session_state["tab7_loaded_subtabs"] = [name.split(":")[1]]
        at.run()
        exceptions = [e.value[:300] for e in at.exception]
        errors = [str(e.value)[:200] for e in at.error]
        results[name] = {"exceptions": exceptions, "errors": errors, "seconds": round(time.perf_counter() - started, 1),
                         "widgets": len(at.button) + len(at.selectbox) + len(at.checkbox) + len(at.text_input)}
    except Exception as exc:
        results[name] = {"exceptions": [f"harness: {type(exc).__name__}: {exc}"[:300]], "errors": [], "seconds": 0, "widgets": 0}
    print(name, "exceptions:", len(results[name]["exceptions"]), "errors:", len(results[name]["errors"]),
          f"{results[name]['seconds']}s", flush=True)
OUT.write_text(json.dumps(results, indent=1))
