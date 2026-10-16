"""SPACEMAP Viewer: image sources, tiles, cell and annotation layers, catalog and routes."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from spacemap.viewer import annotations, catalog, cells, render
from spacemap.viewer.sources import ImageSource, open_source


# ---------------------------------------------------------------- synthetic images

def _write_channel_pages(path, channels=3, height=1200, width=1000):
    """The layout SPACEMAP writes: one page per channel, lower levels as SubIFDs."""
    names = [f"M{i}" for i in range(channels)]
    ome = ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
           '<Image ID="Image:0" Name="Merged Image"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" '
           f'SizeX="{width}" SizeY="{height}" SizeC="{channels}" SizeZ="1" SizeT="1" PhysicalSizeX="0.5">'
           + "".join(f'<Channel ID="Channel:0:{i}" Name="{n}" SamplesPerPixel="1"/>' for i, n in enumerate(names))
           + "".join(f'<TiffData FirstC="{i}" IFD="{i}" PlaneCount="1"/>' for i in range(channels))
           + "</Pixels></Image></OME>")
    base = np.arange(height * width, dtype=np.uint32).reshape(height, width) % 4000
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for c in range(channels):
            data = (base + c * 1000).astype(np.uint16)
            writer.write(data, subifds=2, tile=(256, 256), photometric="minisblack",
                         description=ome if c == 0 else None, metadata=None)
            writer.write(data[::2, ::2], subfiletype=1, tile=(256, 256), photometric="minisblack")
            writer.write(data[::4, ::4], subfiletype=1, tile=(256, 256), photometric="minisblack")
    return names


def _write_qptiff(path, channels=("DAPI", "CD8"), height=800, width=640):
    """What the scanner writes: full-size channel pages, a thumbnail, then each halved level."""
    data = (np.random.default_rng(0).random((len(channels), height, width)) * 4000).astype(np.uint16)
    with tifffile.TiffWriter(path) as writer:
        def pages(stack, names):
            for plane, name in zip(stack, names):
                description = f"<PerkinElmer-QPI-ImageDescription><Biomarker>{name}</Biomarker></PerkinElmer-QPI-ImageDescription>"
                writer.write(plane, tile=(256, 256), software="PerkinElmer-QPI", description=description, metadata=None)
        pages(data, channels)
        writer.write(data[0, ::8, ::8], software="PerkinElmer-QPI", metadata=None)  # thumbnail
        pages(data[:, ::2, ::2], channels)
    return data


def _write_rgb_pyramid(path):
    data = (np.random.default_rng(0).random((800, 640, 3)) * 255).astype(np.uint8)
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, subifds=1, tile=(256, 256), photometric="rgb")
        writer.write(data[::2, ::2], subfiletype=1, tile=(256, 256), photometric="rgb")
    return data


def test_spacemap_ome_tiff_is_read_page_by_page(tmp_path):
    path = tmp_path / "roi.ome.tiff"
    names = _write_channel_pages(path)
    source = ImageSource(path)
    info = source.info
    assert info.layout == "channel-pages"
    assert info.channel_names == names
    assert (info.width, info.height) == (1000, 1200)
    assert len(info.levels) == 3 and info.levels[1].downsample_x == 2
    assert info.pixel_size_um == 0.5
    full = source.read_region(0, 1, 0, 0, 1000, 1200)
    np.testing.assert_array_equal(source.read_region(1, 1, 10, 20, 30, 40), full[40:80:2, 20:60:2])


def test_raw_qptiff_scans_and_rgb_images(tmp_path):
    scan = tmp_path / "P1_codex.qptiff"
    data = _write_qptiff(scan)
    source = open_source(scan)
    info = source.info
    assert info.layout == "series" and not info.rgb
    assert info.channel_names == ["DAPI", "CD8"] and len(info.levels) == 2
    np.testing.assert_array_equal(source.read_region(0, 1, 5, 5, 25, 15), data[1, 5:15, 5:25])
    np.testing.assert_array_equal(source.read_region(1, 0, 0, 0, 10, 10), data[0, :20:2, :20:2])
    colour = tmp_path / "he.tif"
    _write_rgb_pyramid(colour)
    rgb = open_source(colour)
    assert rgb.info.rgb and rgb.read_region(1, 0, 0, 0, 16, 16).shape == (16, 16, 3)


def test_tiles_blend_channels_and_clip_at_the_edge(tmp_path):
    path = tmp_path / "roi.ome.tiff"
    _write_channel_pages(path)
    info = open_source(path).info
    styles = [render.ChannelStyle(0, (1.0, 0.0, 0.0), 0, 4000)]
    tile = Image.open(io.BytesIO(render.render_tile(str(path), info, 0, 1, 2, styles)))
    assert tile.size == (1000 - 512, 1200 - 1024)  # edge tile is clipped, not padded
    pixels = np.asarray(tile.convert("RGB")).astype(int)
    assert pixels[..., 0].max() > 200 and pixels[..., 1:].max() < 40  # only the red channel is drawn
    overview = Image.open(io.BytesIO(render.render_tile(str(path), info, render.max_zoom(info), 0, 0, styles)))
    assert max(overview.size) <= render.TILE
    low, high = render.channel_limits(str(path), info, 0)
    assert 0 <= low < high


# ---------------------------------------------------------------- cells

def _square(label, x, y, size=20):
    ring = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    return {"type": "Feature", "properties": {"label": label}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


@pytest.fixture
def roi_project(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    image = tmp_path / "s1r1.ome.tiff"
    _write_channel_pages(image, channels=2, height=600, width=600)
    seg = tmp_path / "s1r1.ome_cell_boundaries.geojson"
    features = [_square(i + 1, 40 * (i % 10) + 10, 40 * (i // 10) + 10) for i in range(40)]
    seg.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    table = tmp_path / "unified.csv"
    rows = pd.DataFrame({
        "batch_id": ["s1r1"] * 40 + ["s9r9"] * 5,
        "label": list(range(1, 41)) + list(range(1, 6)),
        "M0": np.linspace(0, 100, 45),
        "phenotype_leiden": (["T"] * 20 + ["B"] * 20) + ["X"] * 5,
    })
    rows.to_csv(table, index=False)
    pyramid = cells.ensure_label_pyramid(seg, 600, 600, tmp_path / "cache")
    return {"image": str(image), "seg": seg, "table": str(table), "pyramid": str(pyramid), "root": tmp_path}


def test_label_pyramid_matches_the_outlines_and_is_reused(roi_project):
    info = open_source(roi_project["image"]).info
    labels = cells.label_tile(roi_project["pyramid"], info, 0, 0, 0)
    assert labels[15, 15] == 1 and labels[15, 55] == 2 and labels[0, 0] == 0
    stamp = cells.label_pyramid_path(roi_project["root"] / "cache", roi_project["seg"]).stat().st_mtime_ns
    cells.ensure_label_pyramid(roi_project["seg"], 600, 600, roi_project["root"] / "cache")
    assert cells.label_pyramid_path(roi_project["root"] / "cache", roi_project["seg"]).stat().st_mtime_ns == stamp


def test_cells_are_coloured_by_a_column_and_classes_can_be_hidden(roi_project):
    labels, values = cells.roi_values(roi_project["table"], "s1r1", "phenotype_leiden")
    assert len(labels) == 40  # only this ROI's cells
    lut, legend = cells.categorical_lut(labels, values, colors={"T": "#ff0000"}, hidden={"B"})
    assert {item["value"] for item in legend} == {"T", "B"}
    assert tuple(lut[1]) == (255, 0, 0, 255) and lut[25][3] == 0
    info = open_source(roi_project["image"]).info
    tile = cells.render_cell_tile(cells.label_tile(roi_project["pyramid"], info, 0, 0, 0), lut, mode="fill")
    assert tuple(tile[15, 15]) == (255, 0, 0, 255)
    numeric_lut, scale = cells.numeric_lut(*cells.roi_values(roi_project["table"], "s1r1", "M0"))
    assert scale["max"] > scale["min"] and numeric_lut[40][3] == 255


def test_labelling_cells_saves_to_the_label_store_and_can_be_undone(roi_project):
    from spacemap import dataio

    result = cells.assign(roi_project["table"], "s1r1", [1, 2, 99], "phenotype_ref1", "CD8 T")
    assert result["assigned"] == 2
    assert "phenotype_ref1" in dataio.columns(roi_project["table"])  # every tab can pick the new column
    frame = dataio.load(roi_project["table"], columns=["batch_id", "label", "phenotype_ref1"])
    labelled = frame[frame["phenotype_ref1"] == "CD8 T"]
    assert set(zip(labelled["batch_id"], labelled["label"])) == {("s1r1", 1), ("s1r1", 2)}
    again = cells.assign(roi_project["table"], "s1r1", [1], "phenotype_ref1", "NK")
    assert again["previous"] == {"1": "CD8 T"}
    cells.assign_many(roi_project["table"], "s1r1", again["previous"], "phenotype_ref1")
    _, values = cells.roi_values(roi_project["table"], "s1r1", "phenotype_ref1")
    assert values[0] == "CD8 T"


def test_cell_details_come_from_the_fast_copy(roi_project, monkeypatch):
    from spacemap import dataio

    details = cells.cell_details(roi_project["table"], "s1r1", 3)
    assert "_note" in details  # no fast copy yet: says so instead of parsing the whole CSV
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    assert dataio.ensure_fast_copy(roi_project["table"], force=True)
    cells.assign(roi_project["table"], "s1r1", [3], "phenotype_ref1", "Treg")
    details = cells.cell_details(roi_project["table"], "s1r1", 3)
    assert details["phenotype_leiden"] == "T" and details["phenotype_ref1"] == "Treg"


# ---------------------------------------------------------------- annotations

def test_annotations_are_drawn_in_class_colours(tmp_path):
    feature = _square(0, 100, 100, 200)
    feature["properties"] = {"classification": {"name": "tumor", "color": [255, 0, 0]}}
    collection = {"type": "Feature", "properties": {"classification": {"name": "stroma"}},
                  "geometry": {"type": "GeometryCollection", "geometries": [_square(0, 400, 400, 50)["geometry"]]}}
    path = tmp_path / "s1r1_tissue_classes.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature, collection]}))
    shapes = annotations.load_annotations(path)
    assert {item["value"] for item in shapes.legend()} == {"tumor", "stroma"}
    image = tmp_path / "s1r1.ome.tiff"
    _write_channel_pages(image, channels=1, height=600, width=600)
    tile = annotations.render_annotation_tile(shapes, open_source(image).info, 0, 0, 0)
    assert tuple(tile[200, 200, :3]) == (255, 0, 0) and tile[200, 200, 3] > 0
    assert tile[425, 425, 3] > 0  # geometry collections are drawn too
    hidden = annotations.render_annotation_tile(shapes, open_source(image).info, 0, 0, 0, hidden={"tumor"})
    assert hidden[200, 200, 3] == 0


# ---------------------------------------------------------------- catalog

@pytest.mark.parametrize("name, sample", [
    ("P11_codex.qptiff", "P11"), ("P11_he.qptiff", "P11"), ("P11_merged.ome.tiff", "P11"),
    ("s4r2.ome.tiff", "s4r2"), ("s4r2.ome_cell_boundaries.geojson", "s4r2"),
    ("s10r1.ome.tiff - Merged Image.geojson", "s10r1"),
    ("s4r2_all_good.ome.tiff - Merged Image_merged.geojson", "s4r2"),
    ("s1r1_tissue_classes.geojson", "s1r1"), ("s10r1_merged.geojson", "s10r1"),
])
def test_file_names_link_to_their_sample(name, sample):
    assert catalog.sample_of(name) == sample


def test_catalog_finds_images_and_what_belongs_to_them(tmp_path):
    for folder, name in (("data/codex", "P1_codex.qptiff"), ("outputs/rois", "s1r1.ome.tiff"),
                         ("outputs/segmentation/geojson", "s1r1.ome_cell_boundaries.geojson"),
                         ("outputs/annotations", "s1r1_all_good.ome.tiff - Merged Image_merged.geojson"),
                         ("outputs/annotations", "s2r1_merged.geojson")):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        content = {"type": "FeatureCollection", "features": [_square(1, 0, 0)]} if "cell_boundaries" in name else {}
        (tmp_path / folder / name).write_text(json.dumps(content))
    entries = catalog.list_images(tmp_path, {})
    assert {(e.group, e.sample) for e in entries} == {("Raw CODEX", "P1"), ("ROIs", "s1r1")}
    roi = next(e for e in entries if e.sample == "s1r1")
    bundle = catalog.bundle_for(tmp_path, {}, roi, None)
    assert bundle.segmentation.endswith("s1r1.ome_cell_boundaries.geojson")
    assert [label for label, _ in bundle.annotations] == ["outputs/annotations / s1r1_all_good.ome.tiff - Merged Image_merged.geojson"]


# ---------------------------------------------------------------- routes

@pytest.fixture
def client():
    from starlette.testclient import TestClient

    from spacemap_gui.viewer import api, registry

    registry.clear()
    return TestClient(api.app)


def test_routes_serve_only_registered_images(client, tmp_path):
    from spacemap_gui.viewer import registry

    path = tmp_path / "roi.ome.tiff"
    _write_channel_pages(path)
    token = registry.register(path)
    info = client.get(f"/image/{token}/info").json()
    assert info["channels"] == ["M0", "M1", "M2"] and "path" not in info
    assert client.get(f"/image/{token}/limits?ch=0,1").json().keys() == {"0", "1"}
    tile = client.get(f"/image/{token}/tile/0/0/0.jpg?s=0:ff0000:0:4000;1:00ff00:0:4000")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg"
    assert client.get("/image/not-a-token/info").status_code == 404
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image")
    with pytest.raises(ValueError):
        registry.register(notes)


def test_cell_and_annotation_routes(client, roi_project):
    from spacemap_gui.viewer import registry

    token = registry.register_object("cells", image=roi_project["image"], pyramid=roi_project["pyramid"],
                                     roi="s1r1", tables={"t0": roi_project["table"]}, colors={})
    assert "phenotype_leiden" in client.get(f"/cells/{token}/columns?t=t0").json()["columns"]
    legend = client.get(f"/cells/{token}/legend?t=t0&col=phenotype_leiden").json()
    assert legend["kind"] == "categorical" and {i["value"] for i in legend["items"]} == {"T", "B"}
    tile = client.get(f"/cells/{token}/tile/0/0/0.png?mode=both&t=t0&col=phenotype_leiden")
    assert tile.status_code == 200 and Image.open(io.BytesIO(tile.content)).mode == "RGBA"
    assert client.get(f"/cells/{token}/at?x=15&y=15&t=t0").json()["label"] == 1
    clicked = client.post(f"/cells/{token}/assign", json={"t": "t0", "column": "phenotype_ref1", "value": "B cell",
                                                          "points": [[15, 15]]}).json()
    assert clicked["assigned"] == 1
    boxed = client.post(f"/cells/{token}/assign", json={"t": "t0", "column": "phenotype_ref1", "value": "Mac",
                                                        "rect": [0, 0, 80, 40]}).json()
    assert boxed["assigned"] == 2 and boxed["previous"]["1"] == "B cell"
    client.post(f"/cells/{token}/assign", json={"t": "t0", "column": "phenotype_ref1", "values": boxed["previous"]})
    legend = client.get(f"/cells/{token}/legend?t=t0&col=phenotype_ref1").json()
    assert {i["value"]: i["count"] for i in legend["items"]} == {"B cell": 1}
    assert client.post(f"/cells/{token}/assign", json={"t": "t0", "column": "", "labels": [1]}).status_code == 400

    path = roi_project["root"] / "s1r1_tissue_classes.geojson"
    feature = _square(0, 0, 0, 300)
    feature["properties"] = {"classification": {"name": "tumor"}}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    ann = registry.register_object("annotations", path=str(path), image=roi_project["image"])
    assert client.get(f"/annotations/{ann}/legend").json()["items"][0]["value"] == "tumor"
    assert client.get(f"/annotations/{ann}/tile/0/0/0.png").status_code == 200


def test_the_package_loads_its_heavy_modules_only_on_use():
    import subprocess
    import sys

    probe = ("import sys, spacemap; heavy = [m for m in ('torch', 'cellpose', 'scimap') if m in sys.modules]; "
             "print(','.join(heavy) or 'none'); spacemap.config")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "none"


def test_each_tab_opens_the_viewer_on_its_own_results():
    unified = ("Unified cell table", ["batch_id", "label", "phenotype_leiden", "phenotype_ref1",
                                      "phenotype_ml_consensus"])
    neighbourhoods = ("Spatial analysis / neighborhood_cell_level.csv", ["label", "phenotype_leiden", "CN_cluster",
                                                                           "neighborhood"])
    assert catalog.suggest_colouring("tab7", [unified, neighbourhoods]) == (0, "phenotype_ml_consensus")
    assert catalog.suggest_colouring("tab8", [unified, neighbourhoods]) == (1, "neighborhood")
    assert catalog.suggest_colouring("tab5", [unified]) == (0, "")
    assert catalog.suggest_colouring("tab9", []) == (0, "")
    assert catalog.start_group("tab1") == "Registered" and catalog.start_group("tab7") == "ROIs"


def test_the_launcher_serves_the_viewer_routes_on_the_same_port():
    import streamlit

    from spacemap_gui import launch

    expected = "server.py" if hasattr(streamlit, "App") else "app.py"
    assert launch.app_path().name == expected and launch.app_path().is_file()


def test_the_pop_up_opens_on_a_project_roi_from_any_tab(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    (tmp_path / "outputs" / "rois").mkdir(parents=True)
    (tmp_path / "outputs" / "segmentation" / "geojson").mkdir(parents=True)
    (tmp_path / "outputs" / "batch_corrected").mkdir(parents=True)
    _write_channel_pages(tmp_path / "outputs" / "rois" / "s1r1.ome.tiff", channels=2, height=300, width=300)
    features = [_square(i + 1, 30 * i + 5, 20) for i in range(5)]
    (tmp_path / "outputs" / "segmentation" / "geojson" / "s1r1.ome_cell_boundaries.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}))
    pd.DataFrame({"batch_id": ["s1r1"] * 5, "label": range(1, 6), "phenotype_leiden": list("AABBC")}).to_csv(
        tmp_path / "outputs" / "batch_corrected" / "unified.csv", index=False)

    def page(root):
        import streamlit as st

        from spacemap_gui.viewer import ui

        st.session_state["active_project_root"] = root
        st.session_state["active_project"] = {"unified_csv_filename": "unified.csv", "directory_mappings": {}}
        ui.render_viewer_launcher("tab7")

    app = AppTest.from_function(page, args=(str(tmp_path),), default_timeout=120)
    app.run()
    app.button(key="tab7_open_image_viewer_btn").click().run()
    assert not app.exception, app.exception
    assert app.selectbox[0].value.endswith("s1r1.ome.tiff")
    outlines = tmp_path / "outputs" / "viewer_cache" / "cell_labels" / "s1r1.ome_cell_boundaries.cell_labels.tif"
    for _ in range(200):  # prepared in the background
        if cells.pyramid_ready(outlines):
            break
        time.sleep(0.1)
    assert cells.pyramid_ready(outlines)


def test_flat_channels_are_not_shown_black(tmp_path):
    path = tmp_path / "mask.tif"
    mask = np.zeros((600, 600), np.uint16)
    mask[100:200, 100:200] = 3000
    tifffile.imwrite(path, mask, tile=(256, 256))
    info = open_source(path).info
    low, high = render.channel_limits(str(path), info, 0)
    assert low <= 0 and high >= 3000


# ---------------------------------------------------------------- shapes, display settings, histograms

def _circle(cx, cy, r, n):
    import math

    ring = [[cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)] for i in range(n)]
    return {"type": "Feature", "properties": {"name": "big"}, "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}


def test_shapes_keep_full_detail_when_only_a_simplified_copy_was_shown(tmp_path):
    from spacemap.viewer import shapes

    path = tmp_path / "rois.geojson"
    saved = shapes.save(path, [_circle(500, 500, 400, 20000), _square(0, 10, 10, 50)])
    assert [f["properties"]["label"] for f in saved] == ["big", "Shape 2"]  # Tab 3 reads "label"
    shown = shapes.for_display(saved, max_vertices=500)
    assert shown[0]["properties"]["simplified"] and len(shown[0]["geometry"]["coordinates"][0]) <= 500
    shown[0]["properties"]["name"] = "Tumour core"  # renamed in the viewer
    del shown[1]  # deleted in the viewer
    kept = shapes.save(path, shapes.merge_edits(shapes.load(path), shown))
    assert len(kept) == 1 and kept[0]["properties"]["name"] == "Tumour core"
    assert len(kept[0]["geometry"]["coordinates"][0]) > 19000 and "simplified" not in kept[0]["properties"]
    assert shapes.mask(kept, 1000, 1000, downsample=10)[50, 50]


def test_marker_settings_give_thresholds_and_new_names(tmp_path):
    from spacemap.viewer import display

    path = tmp_path / "display" / "roi.json"
    display.save(path, [{"index": 0, "name": "DAPI", "color": "#4F7DFF", "low": 120, "high": 900, "visible": True},
                        {"index": 1, "name": "CD8", "label": "CD8a", "color": "red", "low": None, "high": "x"},
                        {"name": "no index"}])
    loaded = display.load(path)["channels"]
    assert [c["color"] for c in loaded] == ["#4f7dff", "#ffffff"]
    assert display.minimums(path) == {"DAPI": 120.0} and display.renamed(path) == {"CD8": "CD8a"}
    assert display.load(tmp_path / "missing.json") == {"channels": []}


def test_histograms_for_the_whole_image_and_inside_a_shape(tmp_path):
    from shapely.geometry import box

    path = tmp_path / "roi.ome.tiff"
    _write_channel_pages(path, channels=1, height=600, width=600)
    info = open_source(path).info
    whole = render.channel_histogram(str(path), info, 0)
    assert sum(whole["counts"]) == whole["pixels"] and whole["axis"][1] <= whole["max"]
    inside = render.channel_histogram(str(path), info, 0, region=box(10, 10, 20, 12))
    assert inside["downsample"] == 1 and inside["pixels"] <= 11 * 3 and inside["max"] < 4000
    with pytest.raises(ValueError):
        render.channel_histogram(str(path), info, 0, region=box(700, 700, 800, 800))


def test_shape_display_and_histogram_routes(client, tmp_path):
    from spacemap_gui.viewer import registry

    image = tmp_path / "roi.ome.tiff"
    _write_channel_pages(image, channels=2, height=600, width=600)
    token = registry.register(image)
    other = tmp_path / "qupath.geojson"
    other.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(0, 100, 100, 80)]}))
    target = tmp_path / "viewer" / "shapes" / "roi.geojson"
    layer = registry.register_object("shapes", path=str(target), image=str(image), editable=True,
                                     sources=[["QuPath export", str(other)]])
    initial = client.get(f"/shapes/{layer}").json()
    assert initial["features"] == [] and initial["editable"] is True
    from spacemap_gui.studio.annotation_edits import revision
    assert initial["revision"] == revision(target)
    drawn = {"type": "Feature", "properties": {"id": "a1", "name": "ROI 1"},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]]}}
    assert client.post(f"/shapes/{layer}/save", json={"features": [drawn]}).json() == {"count": 1}
    assert client.get(f"/shapes/{layer}/sources").json() == {"sources": [{"id": 0, "name": "QuPath export"}]}
    imported = client.post(f"/shapes/{layer}/import", json={"source": 0}).json()
    assert imported["added"] == 1 and imported["count"] == 2
    uploaded = client.post(f"/shapes/{layer}/import", json={"features": [_square(0, 300, 300, 20)]}).json()
    assert uploaded["count"] == 3
    download = client.get(f"/shapes/{layer}/download")
    assert "attachment" in download.headers["content-disposition"] and len(download.json()["features"]) == 3
    histogram = client.get(f"/image/{token}/histogram?ch=1&shape={layer}&sid=a1").json()
    assert histogram["pixels"] > 0 and histogram["downsample"] == 1
    assert client.get(f"/image/{token}/histogram?ch=1&shape={layer}&sid=gone").status_code == 404
    # A layer in pieces is measured as one region: all of it, all of one name, or several picked polygons.
    whole = client.get(f"/image/{token}/histogram?ch=1&shape={layer}").json()
    named = client.get(f"/image/{token}/histogram", params={"ch": 1, "shape": layer, "name": "ROI 1"}).json()
    ids = [str(f["properties"]["id"]) for f in download.json()["features"]]
    picked = client.get(f"/image/{token}/histogram", params={"ch": 1, "shape": layer, "sid": ",".join(ids[:2])}).json()
    assert named["pixels"] == histogram["pixels"] < picked["pixels"] <= whole["pixels"]
    assert client.get(f"/image/{token}/histogram", params={"ch": 1, "shape": layer, "name": "Nobody"}).status_code == 404
    assert client.get(f"/image/{token}/background", params={"ch": 1, "shape": layer}).json()["pixels"] == whole["pixels"]

    read_only = registry.register_object("shapes", path=str(target), image=str(image), editable=False, sources=[])
    assert client.post(f"/shapes/{read_only}/save", json={"features": []}).status_code == 403
    assert len(json.loads(target.read_text())["features"]) == 3

    settings = registry.register_object("display", path=str(tmp_path / "display.json"), image=str(image))
    assert client.get(f"/display/{settings}").json() == {"channels": []}
    client.post(f"/display/{settings}/save", json={"channels": [{"index": 1, "name": "M1", "low": 50, "color": "#00ff00"}]})
    assert client.get(f"/display/{settings}").json()["channels"][0]["low"] == 50.0


def test_tab3_opens_the_viewer_on_its_roi_file(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    (tmp_path / "outputs" / "registered").mkdir(parents=True)
    (tmp_path / "data" / "boundaries").mkdir(parents=True)
    _write_channel_pages(tmp_path / "outputs" / "registered" / "P1_merged.ome.tiff", channels=2, height=300, width=300)

    def page(root):
        from pathlib import Path

        import streamlit as st

        from spacemap_gui.tabs import tab3_roi_extraction

        st.session_state["active_project_root"] = root
        st.session_state["active_project"] = {"directory_mappings": {}}
        st.session_state.setdefault("tab3_input_dir", str(Path(root) / "outputs" / "registered"))
        st.session_state.setdefault("tab3_geojson_dir", str(Path(root) / "data" / "boundaries"))
        tab3_roi_extraction.render_single_image(Path(root))

    app = AppTest.from_function(page, args=(str(tmp_path),), default_timeout=180)
    app.run()
    next(box for box in app.selectbox if box.label == "Select Image").select("P1_merged.ome.tiff").run()
    assert not app.exception, app.exception
    app.button(key="tab3_viewer_draw_btn").click().run()
    assert not app.exception, app.exception
    preset = app.session_state["_viewer_preset"]
    rois = next(layer for layer in preset["layers"] if layer["type"] == "shapes")
    assert rois["name"] == "ROIs" and rois["classes"][0]["name"] == "ROI"
    from spacemap_gui.viewer import registry

    target = registry.resolve_object(rois["token"], "shapes")["path"]
    assert target.endswith("data/boundaries/P1_roi_bounding_boxes_coords.geojson")


def test_roi_boundary_files_are_never_taken_for_cell_segmentation(tmp_path):
    roi_boxes = tmp_path / "data" / "annotations" / "new"  # Tab 3's folder, mapped as batch_geojson_dir
    outlines = tmp_path / "outputs" / "segmentation" / "geojson"
    roi_boxes.mkdir(parents=True)
    outlines.mkdir(parents=True)
    box = {"type": "Feature", "properties": {"name": "s4r1"}, "geometry": _square(0, 0, 0, 900)["geometry"]}
    (roi_boxes / "s4r1.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [box]}))
    (outlines / "s4r1.ome_cell_boundaries.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [_square(1, 5, 5)]}))
    mappings = {"batch_geojson_dir": str(roi_boxes), "tab4_geojson_dir": str(tmp_path / "missing")}
    found = catalog.find_segmentation(tmp_path, mappings, "s4r1")
    assert found and found.endswith("s4r1.ome_cell_boundaries.geojson")
    (outlines / "s4r1.ome_cell_boundaries.geojson").unlink()
    assert catalog.find_segmentation(tmp_path, mappings, "s4r1") is None  # ROI boxes are not cells


def test_an_empty_outline_image_is_reported(tmp_path):
    source = tmp_path / "s1r1_rois.geojson"
    source.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "box"}, "geometry": _square(0, 0, 0, 50)["geometry"]}]}))
    pyramid = cells.ensure_label_pyramid(source, 100, 100, tmp_path / "cache")
    assert cells.pyramid_cell_count(pyramid) == 0
    stamp = pyramid.stat().st_mtime_ns
    assert cells.ensure_label_pyramid(source, 100, 100, tmp_path / "cache").stat().st_mtime_ns == stamp  # reused


# ---------------------------------------------------------------- density maps, added files, viewer windows

def test_density_maps_count_chosen_cells_and_average_columns(roi_project):
    import pandas as pd

    table = roi_project["table"]
    frame = pd.read_csv(table)
    frame["x_centroid"] = [40 * (i % 10) + 20 for i in range(40)] + [0] * 5
    frame["y_centroid"] = [40 * (i // 10) + 20 for i in range(40)] + [0] * 5
    frame.to_csv(table, index=False)
    info = open_source(roi_project["image"]).info
    everyone = cells.density_grid(table, "s1r1", info, radius_um=20)
    only_t = cells.density_grid(table, "s1r1", info, filter_column="phenotype_leiden", filter_values=("T",), radius_um=20)
    assert everyone["cells"] == 40 and only_t["cells"] == 20 and everyone["kind"] == "density"
    assert np.nansum(everyone["grid"]) > np.nansum(only_t["grid"]) > 0
    average = cells.density_grid(table, "s1r1", info, value_column="M0", radius_um=20)
    assert average["kind"] == "average" and average["units"] == "M0"
    tile = cells.render_density_tile(everyone, info, 0, 0, 0)
    assert tile.shape == (512, 512, 4) and tile[60, 100, 3] > 0 and tile[500, 500, 3] == 0  # cells vs empty corner


def test_density_stamp_added_files_and_window_routes(client, roi_project, tmp_path):
    import pandas as pd

    from spacemap_gui.viewer import registry

    frame = pd.read_csv(roi_project["table"])
    frame["x_centroid"], frame["y_centroid"] = 100.0, 100.0
    frame.to_csv(roi_project["table"], index=False)
    token = registry.register_object("cells", image=roi_project["image"], pyramid=roi_project["pyramid"],
                                     roi="s1r1", tables={"t0": roi_project["table"]}, colors={})
    query = "t=t0&fcol=phenotype_leiden&fvals=T&r=30&cmap=magma"
    assert client.get(f"/cells/{token}/density/0/0/0.png?{query}").headers["content-type"] == "image/png"
    legend = client.get(f"/cells/{token}/density/legend?{query}").json()
    assert legend["cells"] == 20 and legend["colormap"] == "magma"
    assert client.get(f"/cells/{token}/density/legend?t=missing").status_code == 400
    before = client.get(f"/cells/{token}/stamp?t=t0").json()["stamp"]
    cells.assign(roi_project["table"], "s1r1", [1], "phenotype_ref1", "NK")
    assert client.get(f"/cells/{token}/stamp?t=t0").json()["stamp"] != before  # labels count as a change

    project = tmp_path / "PROJ"
    (project / "outputs" / "zones").mkdir(parents=True)
    zone = {"type": "FeatureCollection", "features": [_square(0, 10, 10, 100)]}
    (project / "outputs" / "zones" / "s1r1_zones.geojson").write_text(json.dumps(zone))
    workspace = registry.register_object("workspace", root=str(project), imports=str(project / "outputs" / "viewer" / "imported"),
                                         image=roi_project["image"])
    assert client.get(f"/workspace/{workspace}/files").json()["files"] == ["outputs/zones/s1r1_zones.geojson"]
    added = client.post(f"/workspace/{workspace}/add", json={"relative": "outputs/zones/s1r1_zones.geojson"}).json()
    assert added["shapes"] == 1 and added["layer"]["type"] == "annotations"
    assert client.get(f"/annotations/{added['layer']['token']}/tile/0/0/0.png").status_code == 200
    uploaded = client.post(f"/workspace/{workspace}/add", json={"name": "QuPath export.geojson", "geojson": json.dumps(zone)}).json()
    assert uploaded["layer"]["name"] == "Imported / QuPath_export.geojson"
    assert (project / "outputs" / "viewer" / "imported" / "QuPath_export.geojson").is_file()
    assert client.post(f"/workspace/{workspace}/add", json={"relative": "../../etc/passwd"}).status_code == 400
    assert client.post(f"/workspace/{workspace}/add", json={"name": "x", "geojson": "not json"}).status_code == 400

    assert client.get("/session/unknown").status_code == 404
    registry.publish("s1", {"id": "a", "layers": []})
    first = client.get("/session/s1").json()
    assert first["spec"]["id"] == "a" and registry.window_open("s1")
    assert client.get(f"/session/s1?v={first['version']}").json() == {"version": first["version"]}
    registry.publish("s1", {"id": "b", "layers": []})
    assert client.get(f"/session/s1?v={first['version']}").json()["spec"]["id"] == "b"
    assert client.get("/window/index.html").status_code == 200


def test_a_clicked_cell_ranks_its_markers_as_resolve_scores_them(client, roi_project):
    import pandas as pd

    from spacemap_gui.viewer import registry

    frame = pd.read_csv(roi_project["table"])
    rng = np.random.default_rng(1)
    frame["M0"] = rng.random(len(frame)) * 100
    frame["M1"] = rng.random(len(frame)) * 100
    frame.loc[(frame["batch_id"] == "s1r1") & (frame["label"] == 1), ["M0", "M1"]] = [5.0, 900.0]
    frame.to_csv(roi_project["table"], index=False)
    ranking = cells.marker_ranking(roi_project["table"], "s1r1", 1, ["M0", "M1", "missing"])
    assert [item["marker"] for item in ranking] == ["M1", "M0"]  # M1 is far above the ROI's other cells
    roi = frame[frame["batch_id"] == "s1r1"]
    asinh = np.arcsinh(roi["M1"].to_numpy() / 5.0)
    expected_z1 = (np.arcsinh(900.0 / 5.0) - asinh.mean()) / asinh.std()
    assert ranking[0]["z1"] == pytest.approx(expected_z1, rel=1e-4)
    assert ranking[0]["z2"] > 0 > ranking[1]["z2"]  # within the cell, M1 stands out
    token = registry.register_object("cells", image=roi_project["image"], pyramid=roi_project["pyramid"],
                                     roi="s1r1", tables={"t0": roi_project["table"]}, colors={})
    clicked = client.get(f"/cells/{token}/at?x=15&y=15&t=t0").json()
    assert clicked["label"] == 1 and [m["marker"] for m in clicked["ranking"]] == ["M1", "M0"]


def test_triad_density_and_one_colour_maps(client, roi_project):
    import pandas as pd

    from spacemap_gui.viewer import registry

    triads = roi_project["root"] / "outputs" / "spatial_analysis" / "triad"
    triads.mkdir(parents=True)
    rows = [{"batch": "s1r1", "roi": "all", "triad_label": "T::B::Mac" if i % 3 else "T::T::Mac",
             "triad_centroid_x": 50 + i * 5, "triad_centroid_y": 60.0} for i in range(30)]
    rows.append({"batch": "s9r9", "roi": "all", "triad_label": "T::B::Mac", "triad_centroid_x": 10.0, "triad_centroid_y": 10.0})
    pd.DataFrame(rows).to_csv(triads / "triad_instances_all_styles.csv", index=False)
    assert catalog.find_point_tables(roi_project["root"]) == [
        ("Triads / triad_instances_all_styles.csv", str(triads / "triad_instances_all_styles.csv"))]
    table = str(triads / "triad_instances_all_styles.csv")
    info = open_source(roi_project["image"]).info
    grid = cells.density_grid(table, "s1r1", info, filter_column="triad_label", filter_values=("T::B::Mac",), radius_um=20)
    assert grid["cells"] == 20 and grid["units"] == "triads/mm²"
    token = registry.register_object("cells", image=roi_project["image"], pyramid=roi_project["pyramid"], roi="s1r1",
                                     tables={"t0": roi_project["table"], "p0": table}, colors={})
    groups = client.get(f"/cells/{token}/density/groups?t=p0&col=triad_label").json()
    assert {g["value"]: g["count"] for g in groups["items"]} == {"T::B::Mac": 20, "T::T::Mac": 10}
    phenotypes = client.get(f"/cells/{token}/density/groups?t=t0&col=phenotype_leiden").json()
    assert {g["value"] for g in phenotypes["items"]} == {"T", "B"}
    tile = cells.render_density_tile(grid, info, 0, 0, 0, colormap="#00ff00")
    shown = tile[..., 3] > 0
    assert shown.any() and (tile[shown][:, :3] == [0, 255, 0]).all()  # one colour, strength in transparency


def test_a_viewer_window_lists_and_opens_the_projects_images(client, tmp_path, monkeypatch):
    import pandas as pd

    from spacemap_gui.viewer import api, registry

    monkeypatch.setattr(api, "MOUNTED", True)
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    for folder in ("outputs/rois", "outputs/registered", "outputs/segmentation/geojson", "outputs/batch_corrected"):
        (tmp_path / folder).mkdir(parents=True)
    _write_channel_pages(tmp_path / "outputs/rois/s1r1.ome.tiff", channels=2, height=300, width=300)
    _write_channel_pages(tmp_path / "outputs/registered/P1_merged.ome.tiff", channels=2, height=300, width=300)
    (tmp_path / "outputs/segmentation/geojson/s1r1.ome_cell_boundaries.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [_square(i + 1, 30 * i + 5, 20) for i in range(5)]}))
    table = tmp_path / "outputs/batch_corrected/unified.csv"
    pd.DataFrame({"batch_id": ["s1r1"] * 5, "label": range(1, 6), "phenotype_leiden": list("AABBC")}).to_csv(table, index=False)
    registry.publish("win", {"id": "first", "layers": []},
                     project={"root": str(tmp_path), "mappings": {}, "unified": str(table), "colors": {}, "tab": "tab7"})
    images = client.get("/session/win/images").json()["images"]
    assert {(i["group"], i["name"]) for i in images} == {("Registered", "P1_merged.ome.tiff"), ("ROIs", "s1r1.ome.tiff")}
    assert all("path" not in image for image in images)  # the browser never sees paths
    roi = next(i for i in images if i["sample"] == "s1r1")
    opened = client.post("/session/win/open", json={"id": roi["id"]}).json()
    cells_layer = next(layer for layer in opened["spec"]["layers"] if layer["type"] == "cells")
    assert cells_layer["color_by"] == "phenotype_leiden"  # still coloured as the tab it came from asks
    assert client.get("/session/win").json()["spec"]["id"] == opened["spec"]["id"]
    assert client.post("/session/win/open", json={"id": 99}).status_code == 404
    assert client.get("/session/nobody/images").status_code == 404


def test_marker_sets_are_saved_for_the_project_and_listed(client, tmp_path):
    from spacemap_gui.viewer import registry

    image = tmp_path / "roi.ome.tiff"
    _write_channel_pages(image, channels=3)
    display_file = tmp_path / "viewer" / "display" / "roi.ome.tiff.json"
    token = registry.register_object("display", path=str(display_file), image=str(image))
    assert client.get(f"/display/{token}/sets").json() == {"sets": {}}
    t_cells = [{"name": "M1", "label": "CD3", "color": "#FF0000", "low": 10, "high": 500},
               {"name": "M2", "color": "#00ff00", "low": None, "high": None}]
    saved = client.post(f"/display/{token}/sets", json={"name": "T cells", "markers": t_cells}).json()["sets"]
    assert saved["T cells"][0] == {"name": "M1", "label": "CD3", "color": "#ff0000", "low": 10.0, "high": 500.0}
    assert (tmp_path / "viewer" / "display" / "marker_sets.json").is_file()  # one file for the whole project
    client.post(f"/display/{token}/sets", json={"name": "B cells", "markers": [{"name": "M0", "color": "#0000ff"}]})
    assert sorted(client.get(f"/display/{token}/sets").json()["sets"]) == ["B cells", "T cells"]
    assert client.post(f"/display/{token}/sets", json={"name": "", "markers": t_cells}).status_code == 400
    assert client.post(f"/display/{token}/sets", json={"name": "Empty", "markers": []}).status_code == 400
    assert list(client.post(f"/display/{token}/sets", json={"delete": "B cells"}).json()["sets"]) == ["T cells"]


def test_viewer_links_survive_a_restart(tmp_path):
    from spacemap_gui.viewer import registry

    image = tmp_path / "roi.ome.tiff"
    _write_channel_pages(image, channels=1, height=100, width=100)
    registry.clear()
    token = registry.register(image)
    layer = registry.register_object("annotations", path="/x.geojson", image=str(image))
    registry.publish("window-1", {"id": "view", "layers": [{"token": token}]}, project={"root": str(tmp_path)})
    # A restart: memory is empty, the saved table and key are still there.
    for table in (registry._BY_TOKEN, registry._BY_PATH, registry._OBJECTS, registry._OBJECT_KEYS, registry._SESSIONS):
        table.clear()
    registry._STATE.update(loaded=False, secret=None)
    assert registry.resolve(token) == str(image.resolve())
    assert registry.resolve_object(layer, "annotations")["path"] == "/x.geojson"
    assert registry.register(image) == token  # the same image keeps its token
    assert registry.session_view("window-1")["spec"]["id"] == "view"
    assert registry.session_project("window-1") == {"root": str(tmp_path)}
    assert registry.resolve("made-up-token") is None


def test_cell_outlines_are_prepared_in_the_background(client, tmp_path):
    from spacemap_gui.viewer import registry

    image = tmp_path / "s1r1.ome.tiff"
    _write_channel_pages(image, channels=1, height=300, width=300)
    segmentation = tmp_path / "s1r1.ome_cell_boundaries.geojson"
    segmentation.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(1, 5, 5), _square(2, 60, 60)]}))
    pyramid, ready = cells.prepare_label_pyramid(segmentation, 300, 300, tmp_path / "cache")
    assert not ready  # first time: building, the call did not wait
    token = registry.register_object("cells", image=str(image), pyramid=str(pyramid), roi="s1r1", tables={}, colors={})
    for _ in range(200):
        if client.get(f"/cells/{token}/ready").json()["ready"]:
            break
        time.sleep(0.1)
    assert client.get(f"/cells/{token}/ready").json() == {"ready": True, "cells": 2}
    assert cells.prepare_label_pyramid(segmentation, 300, 300, tmp_path / "cache") == (pyramid, True)
    assert client.get(f"/cells/{token}/at?x=10&y=10").json()["label"] == 1


def test_the_browser_channel_is_taken_without_a_fragment_rerun():
    from streamlit.testing.v1 import AppTest

    def page():
        import streamlit as st

        from spacemap_gui.viewer import ui

        st.session_state["first"] = ui._adopt_browser_channel({"event": "browser", "browser": "ab" * 16})
        st.session_state["again"] = ui._adopt_browser_channel({"event": "window", "browser": "ab" * 16})
        st.session_state["bad"] = ui._adopt_browser_channel({"event": "browser", "browser": "../../etc"})

    app = AppTest.from_function(page)
    app.run()
    assert not app.exception
    assert app.session_state["first"] and not app.session_state["again"] and not app.session_state["bad"]
    assert app.session_state["_viewer_session"] == "ab" * 16
    source = (Path(__file__).resolve().parents[1] / "src/spacemap_gui/viewer/ui.py").read_text()
    assert 'scope="fragment"' not in source  # not allowed when the pop-up opens during a full page update


def test_a_folder_shared_by_two_steps_belongs_to_the_later_one(tmp_path):
    rois = tmp_path / "outputs" / "rois"
    subtracted = tmp_path / "outputs" / "background_subtracted"
    rois.mkdir(parents=True)
    subtracted.mkdir(parents=True)
    for folder, name in ((rois, "s1r2.ome.tiff"), (subtracted, "slide4.ome.tiff")):
        (folder / name).write_bytes(b"")
    # PDAC's settings: Tab 2's output points at Tab 3's ROI folder.
    mappings = {"tab2_output_dir": str(rois), "bg_subtraction_output_dir": str(rois)}
    entries = {e.name: e.group for e in catalog.list_images(tmp_path, mappings)}
    assert entries == {"s1r2.ome.tiff": "ROIs", "slide4.ome.tiff": "Background subtracted"}
    assert [e.group for e in catalog.list_images(tmp_path, {})] == ["Background subtracted", "ROIs"]  # pipeline order


def test_images_from_outside_the_project_can_be_added_to_its_list(client, tmp_path, monkeypatch):
    from spacemap_gui.viewer import api, registry

    monkeypatch.setattr(api, "MOUNTED", True)
    project = tmp_path / "project"
    (project / "outputs/rois").mkdir(parents=True)
    _write_channel_pages(project / "outputs/rois/s1r1.ome.tiff", channels=2, height=300, width=300)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    scan = elsewhere / "P11_roi_2.ome.tiff"
    _write_channel_pages(scan, channels=2, height=300, width=300)
    (elsewhere / "notes.txt").write_text("not an image")
    (elsewhere / "broken.ome.tiff").write_bytes(b"not a tiff")

    entry = catalog.add_image(project, f'  "{scan}" ')  # pasted with quotes and spaces
    assert (entry.group, entry.name, entry.sample) == (catalog.ADDED_GROUP, "P11_roi_2.ome.tiff", "P11_roi_2")
    catalog.add_image(project, scan)  # added once only
    assert [e.path for e in catalog.added_images(project)] == [str(scan.resolve())]
    assert (project / "outputs/viewer/added_images.json").is_file()  # kept with the project
    assert catalog.added_images(tmp_path / "other") == []  # each project has its own list
    for bad, reason in ((elsewhere / "missing.tiff", "No file"), (elsewhere / "notes.txt", "not an image"),
                        (elsewhere / "broken.ome.tiff", "could not be read")):
        with pytest.raises(ValueError, match=reason):
            catalog.add_image(project, bad)

    # From a viewer window: the added image is listed and opens; bad paths are refused.
    registry.publish("win", {"id": "first", "layers": []}, project={"root": str(project), "mappings": {}})
    images = client.get("/session/win/images").json()["images"]
    assert [(i["group"], i["name"]) for i in images] == [("ROIs", "s1r1.ome.tiff"),
                                                         (catalog.ADDED_GROUP, "P11_roi_2.ome.tiff")]
    other = elsewhere / "s9r9.ome.tiff"
    _write_channel_pages(other, channels=1, height=200, width=200)
    added = client.post("/session/win/add-image", json={"path": str(other)}).json()
    listed = added["images"]
    assert listed[added["added"]]["name"] == "s9r9.ome.tiff" and all("path" not in i for i in listed)
    again = client.post("/session/win/add-image", json={"path": str(scan)}).json()  # already listed
    assert again["images"][again["added"]]["name"] == "P11_roi_2.ome.tiff" and len(again["images"]) == 3
    opened = client.post("/session/win/open", json={"id": added["added"]}).json()
    assert opened["spec"]["layers"][0]["type"] == "image"
    assert client.post("/session/win/add-image", json={"path": str(elsewhere / "notes.txt")}).status_code == 400
    assert client.post("/session/nobody/add-image", json={"path": str(other)}).status_code == 404

    catalog.remove_image(project, scan)
    assert [e.name for e in catalog.added_images(project)] == ["s9r9.ome.tiff"]
    assert scan.is_file()  # taken off the list only


def test_cell_outlines_are_built_outside_the_interface_process(tmp_path, monkeypatch):
    # A large outline file holds Python's lock while it is read; built in the
    # server itself it froze every page and the viewer for over a minute.
    def not_here(*args, **kwargs):
        raise AssertionError("built inside the interface process")

    monkeypatch.setattr(cells, "build_label_pyramid", not_here)
    monkeypatch.setattr(cells, "ensure_label_pyramid", not_here)
    segmentation = tmp_path / "s2r1.ome_cell_boundaries.geojson"
    segmentation.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(1, 5, 5), _square(7, 40, 40)]}))
    pyramid, ready = cells.prepare_label_pyramid(segmentation, 200, 200, tmp_path / "cache")
    assert not ready
    cells._BUILDING[str(pyramid)].join(120)
    assert cells.pyramid_ready(pyramid) and cells.pyramid_cell_count(pyramid) == 2


def test_the_pop_up_adds_and_removes_an_image_from_elsewhere(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    project = tmp_path / "project"
    (project / "outputs" / "rois").mkdir(parents=True)
    _write_channel_pages(project / "outputs" / "rois" / "s1r1.ome.tiff", channels=2, height=300, width=300)
    scan = tmp_path / "scans" / "P11_roi_2.ome.tiff"
    scan.parent.mkdir()
    _write_channel_pages(scan, channels=2, height=300, width=300)

    def page(root):
        import streamlit as st

        from spacemap_gui.viewer import ui

        st.session_state["active_project_root"] = root
        st.session_state["active_project"] = {"directory_mappings": {}}
        st.session_state[ui._OPEN_KEY] = True  # in the browser the pop-up stays open while it is used
        ui.render_viewer_launcher("tab7")

    app = AppTest.from_function(page, args=(str(project),), default_timeout=120)
    app.run()
    app.text_input(key="viewer_add_image_path").input(str(tmp_path / "scans" / "missing.tiff")).run()
    app.button(key="viewer_add_image_btn").click().run()
    assert app.error and "No file" in app.error[0].value
    app.text_input(key="viewer_add_image_path").input(str(scan)).run()
    app.button(key="viewer_add_image_btn").click().run()
    assert not app.exception, app.exception
    assert not app.warning  # no "value set via Session State" warning
    assert app.selectbox[0].value == f"{catalog.ADDED_GROUP}  |  P11_roi_2.ome.tiff"  # opened at once
    assert f"{catalog.ADDED_GROUP}  |  P11_roi_2.ome.tiff" in app.selectbox[0].options
    app.button(key="viewer_remove_added_image_btn").click().run()
    assert not app.exception, app.exception
    assert app.selectbox[0].options == ["ROIs  |  s1r1.ome.tiff"] and scan.is_file()


def test_the_server_imports_this_checkouts_code_even_with_an_older_install(tmp_path):
    # The cluster environment holds an old editable SPACEMAP (.pth -> another folder).
    import subprocess
    import sys

    decoy = tmp_path / "old" / "spacemap_gui"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("raise ImportError('the old SPACEMAP was imported')")
    server = Path(__file__).resolve().parents[1] / "src" / "spacemap_gui" / "server.py"
    script = ("import runpy, sys; sys.argv = ['x']; import streamlit; streamlit.App = lambda *a, **k: None; "
              f"runpy.run_path({str(server)!r}); import spacemap_gui; print(spacemap_gui.__file__)")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                          cwd=tmp_path, env={**__import__("os").environ, "PYTHONPATH": str(tmp_path / "old")})
    assert done.returncode == 0, done.stderr[-1500:]
    assert done.stdout.strip().splitlines()[-1].startswith(str(server.parents[1]))


def test_live_resolve_in_the_viewer(client, tmp_path, monkeypatch):
    """Open, drag a threshold, colour cells, explain a cell and save, as Tab 7 would assign."""
    from spacemap.viewer import live_resolve
    from spacemap_gui.viewer import registry

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    live_resolve.clear()
    image = tmp_path / "s1r1.ome.tiff"
    _write_channel_pages(image, channels=2, height=400, width=400)
    seg = tmp_path / "s1r1.ome_cell_boundaries.geojson"
    seg.write_text(json.dumps({"type": "FeatureCollection",
                               "features": [_square(i + 1, 40 * (i % 8) + 10, 40 * (i // 8) + 10) for i in range(40)]}))
    pyramid = cells.ensure_label_pyramid(seg, 400, 400, tmp_path / "cache")
    table = tmp_path / "unified.csv"
    cd3 = np.r_[np.full(20, 300.0), np.full(20, 10.0)]
    cd8 = np.r_[np.full(10, 250.0), np.full(10, 5.0), np.full(20, 5.0)]
    pd.DataFrame({"batch_id": ["s1r1"] * 40, "label": range(1, 41), "CD3": cd3, "CD8": cd8,
                  "CD20": np.r_[np.full(20, 1.0), np.full(20, 400.0)]}).to_csv(table, index=False)
    cache = tmp_path / "outputs/phenotyping/cache"
    cache.mkdir(parents=True)
    (cache / "resolve_pheno_panel.json").write_text(json.dumps({"criteria_cols": ["Criteria 1"], "table": [
        {"Phenotype": "T", "Criteria 1": ["CD3"], "Supportive": [], "Must not have": []},
        {"Phenotype": "CD8 T", "Criteria 1": ["CD3", "CD8"], "Supportive": [], "Must not have": []},
        {"Phenotype": "B", "Criteria 1": ["CD20"], "Supportive": [], "Must not have": []}]}))
    thresholds = cache / "resolve_thresholds_table.csv"
    pd.DataFrame([{"batch_id": "s1r1", "Threshold_Source": "Polygon values", "CD3": 100.0, "CD8": 100.0,
                   "CD20": 100.0}, {"batch_id": "s2r1", "Threshold_Source": "x", "CD3": 1, "CD8": 2, "CD20": 3}]
                 ).to_csv(thresholds, index=False)
    from spacemap_gui.viewer import ui

    files = ui.resolve_files(tmp_path, str(table))
    assert files["panel"].endswith("resolve_pheno_panel.json") and files["settings"] is None
    cells_token = registry.register_object("cells", image=str(image), pyramid=str(pyramid), roi="s1r1",
                                           tables={"t0": str(table)}, colors={})
    token = registry.register_object("resolve", cells=cells_token, table=str(table), roi="s1r1",
                                     colors={"CD8_T": "#ff0000"}, **files)

    opened = client.post(f"/resolve/{token}/open", json={}).json()
    assert opened["counts"] == {"CD8 T": 10, "T": 10, "B": 20}
    assert [m["source"] for m in opened["markers"]] == ["saved"] * 3
    changed = client.post(f"/resolve/{token}/thresholds", json={"thresholds": {"CD8": 300.0}}).json()
    assert changed["counts"] == {"T": 20, "B": 20} and changed["version"] == opened["version"] + 1
    tile = client.get(f"/resolve/{token}/tile/0/0/0.png?v={changed['version']}")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    legend = client.get(f"/resolve/{token}/legend").json()["items"]
    assert {item["value"]: item["count"] for item in legend} == {"T": 20, "B": 20}
    histogram = client.get(f"/resolve/{token}/histogram?marker=CD8").json()
    assert histogram["threshold"] == 300.0 and histogram["positive"] == 0 and histogram["total"] == 40
    why = client.get(f"/resolve/{token}/why?x=20&y=20").json()  # cell 1: CD3 300, CD8 250
    assert why["label"] == 1 and why["phenotype"] == "T"
    cd8t = next(p for p in why["phenotypes"] if p["phenotype"] == "CD8 T")
    assert cd8t["criteria"][0]["failing"] == ["CD8"]
    assert client.post(f"/resolve/{token}/save", json={}).json() == {"saved": thresholds.name, "roi": "s1r1"}
    saved = pd.read_csv(thresholds)
    row = saved[saved.batch_id == "s1r1"].iloc[0]
    assert row["CD8"] == 300.0 and row["Threshold_Source"] == live_resolve.SAVED_SOURCE
    assert saved[saved.batch_id == "s2r1"].iloc[0]["CD20"] == 3  # other ROIs untouched
    assert list(cache.glob("resolve_thresholds_table.csv.before-viewer-*"))  # backup kept
    assert client.post(f"/resolve/{token}/thresholds", json={"thresholds": {"Nope": 1}}).status_code == 400
    assert client.get("/resolve/made-up/legend").status_code == 404


def test_rows_of_one_roi_are_read_from_their_row_groups_only(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from spacemap import dataio

    frame = pd.DataFrame({"batch_id": np.repeat(["a", "b", "c", "b"], [10, 13, 9, 5]), "label": np.arange(37),
                          "CD3": np.arange(37) * 1.5})
    twin = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), twin, row_group_size=7)
    monkeypatch.setattr(dataio, "fast_copy_info", lambda path: {"current": True, "twin": str(twin)})
    read = []
    original = pq.ParquetFile.read_row_groups
    monkeypatch.setattr(pq.ParquetFile, "read_row_groups",
                        lambda self, groups, **kw: read.append(list(groups)) or original(self, groups, **kw))
    positions = np.flatnonzero(frame["batch_id"].to_numpy() == "b")  # rows 10-22 and 32-36
    table = cells.fast_rows("t.csv", positions, ["label", "CD3"])
    assert table.column("label").to_pylist() == frame["label"].to_numpy()[positions].tolist()
    assert table.column("CD3").to_pylist() == frame["CD3"].to_numpy()[positions].tolist()
    assert read == [[1, 2, 3, 4, 5]]  # group 0 (rows 0-6, only ROI a) is not read
    assert cells.fast_rows("t.csv", np.array([99]), ["label"]) is None  # beyond the copy: fall back


def test_outlines_are_not_rebuilt_for_another_spelling_of_the_same_folder(tmp_path):
    # Shared storage may have multiple path aliases. The outlines recorded one spelling; the
    # viewer asking with the other rebuilt them (minutes for a large ROI).
    real = tmp_path / "endosome" / "work"
    real.mkdir(parents=True)
    alias = tmp_path / "work"
    alias.symlink_to(real)
    seg = real / "s1r1.ome_cell_boundaries.geojson"
    seg.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(1, 5, 5)]}))
    pyramid = cells.ensure_label_pyramid(alias / seg.name, 100, 100, tmp_path / "cache")
    stamp = pyramid.stat().st_mtime_ns
    assert cells.prepare_label_pyramid(seg, 100, 100, tmp_path / "cache") == (pyramid, True)
    assert cells.ensure_label_pyramid(seg, 100, 100, tmp_path / "cache").stat().st_mtime_ns == stamp
    assert not list((tmp_path / "cache").glob(".*.tmp"))


def test_the_clicked_cell_is_outlined(client, tmp_path, monkeypatch):
    from spacemap_gui.viewer import registry

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    image = tmp_path / "s1r1.ome.tiff"
    _write_channel_pages(image, channels=1, height=300, width=300)
    seg = tmp_path / "s1r1.ome_cell_boundaries.geojson"
    seg.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(1, 40, 60), _square(2, 200, 200)]}))
    pyramid = cells.ensure_label_pyramid(seg, 300, 300, tmp_path / "cache")
    table = tmp_path / "cells.csv"
    pd.DataFrame({"batch_id": ["s1r1", "s1r1"], "label": [1, 2], "x_centroid": [50.0, 210.0],
                  "y_centroid": [70.0, 210.0]}).to_csv(table, index=False)
    token = registry.register_object("cells", image=str(image), pyramid=str(pyramid), roi="s1r1",
                                     tables={"t0": str(table)}, colors={})
    clicked = client.get(f"/cells/{token}/outline?label=1&x=45&y=65").json()
    ring = np.array(clicked["rings"][0])
    assert clicked["label"] == 1 and len(clicked["rings"]) == 1
    assert abs(ring[:, 0].min() - 40) <= 1 and abs(ring[:, 0].max() - 60) <= 1  # the square's edges
    assert abs(ring[:, 1].min() - 60) <= 1 and abs(ring[:, 1].max() - 80) <= 1
    from_table = client.get(f"/cells/{token}/outline?label=2&t=t0").json()  # no click: the table's centroid
    assert np.array(from_table["rings"][0])[:, 0].min() >= 199
    assert client.get(f"/cells/{token}/outline?label=1&x=250&y=20").json()["rings"] == []  # not there
    assert client.get(f"/cells/{token}/outline?x=5").status_code == 400
    big = cells.cell_outline(pyramid, 2, 201, 201, window=4)  # grows the window until the cell fits
    assert abs(np.array(big[0])[:, 0].max() - 220) <= 1


def test_the_studio_and_viewer_buttons_go_in_the_top_bar_columns(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from spacemap_gui.viewer import ui

    monkeypatch.setattr(ui, "studio_link", lambda context: "http://127.0.0.1:1/studio?step=" + context)

    def page():
        import streamlit as st

        from spacemap_gui.viewer import ui

        _, studio, viewer, _ = st.columns([3.2, 1, 1, 1])
        ui.render_viewer_launcher("tab3", studio_area=studio, viewer_area=viewer)

    app = AppTest.from_function(page, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    columns = app.columns
    assert len(columns) == 4
    assert [button.key for button in columns[2].button] == ["tab3_open_image_viewer_btn"]
    assert columns[2].button[0].label == "Image viewer"
