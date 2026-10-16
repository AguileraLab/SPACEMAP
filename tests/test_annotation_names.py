"""An image's annotations by name: the same rule as Layers, Export ticks, export by name, and copies."""
import json
from pathlib import Path

from shapely.geometry import box, mapping

from spacemap_gui.studio import annotation, annotation_edits as edits, annotation_names as names, artifact_learning
from test_studio_regions import _image


def _feature(geometry, **properties):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def test_names_ticks_export_and_copies(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    config = {"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}
    (root / "study.json").write_text(json.dumps(config))
    image = _image(root / "sample.ome.tiff")
    _, output = annotation.folders(root, {})
    sample = annotation.sample_id(image.name)
    tissue = annotation.valid_tissue(output, sample)
    tissue.parent.mkdir(parents=True, exist_ok=True)
    # Tissue in two pieces, one renamed "lymph_node"; it keeps its other piece under the tissue's name.
    edits.replace(tissue, [_feature(box(0, 0, 50, 50)), _feature(box(100, 100, 150, 150), name="lymph_node", label="lymph_node", named=True)],
                  edits.revision(tissue), "test", kind="tissue")
    clots = artifact_learning.polygons_path(output, sample, "clot")
    clots.parent.mkdir(parents=True, exist_ok=True)
    edits.replace(clots, [_feature(box(10, 10, 20, 20), classification={"name": "Blood_Clot", "color": [230, 25, 75]})], edits.revision(clots), "test")
    good = artifact_learning.example_path(root, "clot", image)
    good.parent.mkdir(parents=True, exist_ok=True)
    edits.replace(good, [_feature(box(30, 30, 40, 40), classification={"name": "Good tissue"})], edits.revision(good), "test")

    found = {(g["name"], g["role"]): g for g in names.groups(root, config, image, output)}
    assert found[("tissue", "")]["count"] == 1 and found[("lymph_node", "")]["count"] == 1
    assert found[("Blood clots", "artifact")]["rule"] == "local:Blood clots" and not found[("Blood clots", "good")]["rule"]  # training left out
    assert not any(g["export"] for g in found.values())  # nothing is exported until ticked

    # A copy of the tissue, renamed as a whole: a new name row of the image.
    layer = names.duplicate(root, image, output, "primary", "tissue", [_feature(box(0, 0, 50, 50), name="tissue")])
    assert layer["origin"] == "Copy of tissue" and layer["editable"]
    found = {(g["name"], g["role"]): g for g in names.groups(root, config, image, output)}
    assert found[("primary", "")]["count"] == 1

    # Layers' ticks are the project's rules: ticked here, ticked for every image.
    from spacemap_gui.studio.settings import read_config
    ticks = names.save_choices(root, read_config(root), image, output, {"::tissue::": True, "::Blood clots::artifact": True, "::primary::": True, "::lymph_node::": False})
    assert ticks["::tissue::"] and not ticks["::lymph_node::"]
    config = read_config(root)
    assert names.export_settings(root, config)["rules"]["local:tissue"] == {"export": True}
    result = names.export(root, config, image, output, merge=True)
    assert result["names"] == ["Blood clots", "primary", "tissue"]  # lymph_node unticked, good tissue left out
    exported = {f["properties"]["name"]: f["properties"]["classification"] for f in result["geojson"]["features"]}
    assert exported["Blood clots"]["color"] == [230, 25, 75]


def test_the_name_rule_matches_layers():
    assert names.name_of({"properties": {"name": "tissue_3"}}, "tissue", "tissue") == "tissue"
    assert names.name_of({"properties": {"name": "Tumor", "named": True}}, "regions", "Marker regions · Epithelia") == "Tumor"
    assert names.name_of({"properties": {"name": "Region 4", "classification": {"name": "Marker_Region"}}}, "regions",
                         "Marker regions · Epithelia") == "Epithelia"
    assert names.name_of({"properties": {"classification": {"name": "Stroma"}}}, "regions", "Classification") == "Stroma"
    assert names.name_of({"properties": {"classification": {"name": "Good tissue"}}}, "examples", "Blood clots") == "Blood clots"


def test_layers_leave_the_project_and_files_leave_the_list(tmp_path):
    import pytest

    from spacemap_gui.studio import annotation_inputs

    root = tmp_path / "study"
    root.mkdir()
    config = {"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}
    (root / "study.json").write_text(json.dumps(config))
    image = _image(root / "sample.ome.tiff")
    _, output = annotation.folders(root, {})
    tissue = annotation.valid_tissue(output, annotation.sample_id(image.name))
    tissue.parent.mkdir(parents=True, exist_ok=True)
    edits.replace(tissue, [_feature(box(0, 0, 50, 50))], edits.revision(tissue), "test", kind="tissue")
    copy = names.duplicate(root, image, output, "primary", "tissue", [_feature(box(0, 0, 50, 50))])
    kept = names.remove_layer(root, image, output, copy["saved_file"])
    assert "outputs/studio/deleted" in kept and not any(g["name"] == "primary" for g in names.groups(root, config, image, output))
    with pytest.raises(ValueError, match="delete its polygons"):
        names.remove_layer(root, image, output, tissue)
    found = root / "data/annotations/sample_merged.geojson"  # a file found on disk for this image
    found.parent.mkdir(parents=True)
    found.write_text(json.dumps({"type": "FeatureCollection", "features": [_feature(box(0, 0, 9, 9), classification={"name": "Tumor"})]}))
    assert str(found.resolve()) in annotation_inputs.sources(root, config, image, output)
    names.hide_file(root, image, found)
    assert str(found.resolve()) not in annotation_inputs.sources(root, config, image, output) and found.is_file()


def test_deleting_the_classification_takes_its_named_classes_along(tmp_path):
    from spacemap_gui.studio import annotation, annotation_inputs, annotation_layers, annotation_names

    root = tmp_path / "project"
    output = root / "outputs/annotations"
    output.mkdir(parents=True)
    image = root / "roi.ome.tiff"
    image.write_bytes(b"")
    sample = annotation.sample_id(image.name)
    classes, domains = annotation.classes(output, sample), annotation.domains(output, sample)
    for path in (classes, domains):
        path.write_text('{"type": "FeatureCollection", "features": []}')
    annotation_layers.register(root, image, classes, name="Classification", kind="classes")
    annotation_names.remove_layer(root, image, output, classes)
    assert not classes.exists() and not domains.exists()
    # Named classes left behind by a deletion made before this fix are not listed either.
    domains.write_text('{"type": "FeatureCollection", "features": []}')
    import os, time
    os.utime(domains, (time.time() - 600, time.time() - 600))
    assert annotation_names.stale_domains(root, image, output)
    assert str(domains.resolve()) not in annotation_inputs.sources(root, {}, image, output)


def test_named_classes_follow_the_classification_polygons(tmp_path):
    from shapely.geometry import box, mapping
    from spacemap_gui.studio import annotation, annotation_edits as edits

    output = tmp_path / "out"
    output.mkdir()
    classes, domains = annotation.classes(output, "roi"), annotation.domains(output, "roi")
    feature = lambda name, x: {"type": "Feature", "geometry": mapping(box(x, 0, x + 10, 10)),
                               "properties": {"name": name, "classification": {"name": name}}}
    edits.replace(classes, [feature("Tumor", 0), feature("Stroma", 20)], edits.revision(classes), "Saved Classification")
    assert sorted(f["properties"]["name"] for f in edits.read(domains)[0]) == ["Stroma", "Tumor"]
    found, version = edits.read(classes)
    edits.save(classes, [f for f in found if f["properties"]["name"] == "Stroma"], version)  # Tumor deleted in Layers
    assert [f["properties"]["name"] for f in edits.read(domains)[0]] == ["Stroma"]


def test_a_merged_export_stays_small_and_named_by_the_settings(tmp_path):
    """14,000 polygons of one name export as one annotation, not 14,000 copies of their source list."""
    from pathlib import Path
    import json
    from shapely.geometry import box, mapping
    from spacemap_gui.studio import annotation, annotation_edits as edits, annotation_layers, annotation_names

    root = tmp_path / "project"
    output = root / "outputs/annotations"
    output.mkdir(parents=True)
    image = root / "roi.ome.tiff"
    image.write_bytes(b"")
    layer = output / "roi_marker_regions_stroma.geojson"
    features = [{"type": "Feature", "geometry": mapping(box(x * 3, y * 3, x * 3 + 2, y * 3 + 2)),
                 "properties": {"name": "stroma", "named": True, "classification": {"name": "stroma"}}}
                for x in range(60) for y in range(50)]
    edits.replace(layer, features, edits.revision(layer), "Fixture")
    annotation_layers.register(root, image, layer, name="stroma", kind="regions")
    naming = {**annotation_names.EXPORT_DEFAULTS, "prefix": "QC_", "suffix": "_v1", "rules": {"local:stroma": {"export": True}}}
    result = annotation_names.export(root, {}, image, output, True, naming)
    exported = json.loads((annotation_names.export_folder(root, naming) / "QC_roi_v1.geojson").read_text())
    stroma = [f for f in exported["features"] if f["properties"]["name"] == "stroma"]
    assert len(stroma) == 1 and stroma[0]["properties"]["merged_count"] == 3000 and stroma[0]["properties"]["objectType"] == "annotation"
    assert Path(result["path"]).stat().st_size < 2_000_000 and not (Path(result["path"]).parent / "studio_history").exists()


def test_project_names_newest_first_one_per_name(tmp_path):
    names.remember(tmp_path, {"Normal": "#00FF00"})
    names.remember(tmp_path, {"Tumor": "#ff0000"})
    names.remember(tmp_path, {"normal": "#00aa00"})
    assert names.palette(tmp_path) == [{"name": "normal", "colour": "#00aa00"}, {"name": "Tumor", "colour": "#ff0000"}]
    assert names.remember(tmp_path, forget=["tumor"]) == [{"name": "normal", "colour": "#00aa00"}]
    for bad in ({"": "#000000"}, {"x": "red"}):
        try:
            names.remember(tmp_path, bad)
            raise AssertionError("expected an error")
        except ValueError:
            pass


def _project(tmp_path, *samples):
    """A project with images (in its images folder), each with a saved tissue of 0..40 px."""
    root = tmp_path / "study"
    root.mkdir()
    config = {"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}
    (root / "study.json").write_text(json.dumps(config))
    images_folder, output = annotation.folders(root, {})
    images_folder.mkdir(parents=True, exist_ok=True)
    images = [_image(images_folder / (sample + ".ome.tiff")) for sample in samples]
    for image in images:
        tissue = annotation.valid_tissue(output, annotation.sample_id(image.name))
        tissue.parent.mkdir(parents=True, exist_ok=True)
        edits.replace(tissue, [_feature(box(0, 0, 40, 40))], edits.revision(tissue), "tissue", kind="tissue")
    return root, config, images, output


def _old_file(root, image, *items):
    from spacemap_gui.studio import annotation_inputs

    return annotation_inputs.import_polygons(root, image, annotation.sample_id(image.name) + "_old.geojson",
                                             [_feature(box(*bounds), classification={"name": name, "color": [255, 0, 0]}) for name, bounds in items])


def test_rules_keep_inside_the_imported_tissue_and_subtract_blood_clots(tmp_path):
    """The user's case: an imported tissue is the container, saved Blood clots are subtracted, and imported tumor and
    the imported tissue are exported; the local tissue is not; the files themselves are not changed."""
    from shapely.geometry import shape

    root, config, (image,), output = _project(tmp_path, "A_roi_1")
    sample = annotation.sample_id(image.name)
    clots = artifact_learning.polygons_path(output, sample, "clot")
    edits.replace(clots, [_feature(box(0, 0, 10, 10))], edits.revision(clots), "clots")
    copy = _old_file(root, image, ("tissue", (0, 0, 60, 60)), ("tumor", (50, 50, 80, 80)), ("normal", (0, 0, 20, 20)))
    before = copy.read_bytes()
    rows = {r["rule"]: r for r in names.summary(root, config, [str(image)], output)}
    assert set(rows) == {"local:tissue", "local:Blood clots", "imported:tissue", "imported:tumor", "imported:normal"}
    assert rows["imported:tissue"]["origin"] == "imported" and rows["local:Blood clots"]["role"] == "artifact"
    naming = {**names.EXPORT_DEFAULTS, "inside": "imported:tissue", "rules": {
        "imported:tissue": {"export": True}, "imported:tumor": {"export": True}, "imported:normal": {"export": True, "name": "Normal", "colour": "#00ff00"},
        "local:Blood clots": {"subtract": True}}}
    result = names.export(root, config, image, output, True, naming)
    shapes = {f["properties"]["name"]: f for f in result["geojson"]["features"]}
    assert sorted(shapes) == ["Normal", "tissue", "tumor"] and result["inside"] == "tissue (imported)"
    assert shape(shapes["tissue"]["geometry"]).area == 3600 - 100  # the container minus the clot
    assert shape(shapes["tumor"]["geometry"]).area == 100  # 50..60 inside the imported tissue
    assert shape(shapes["Normal"]["geometry"]).area == 400 - 100 and shapes["Normal"]["properties"]["classification"]["color"] == [0, 255, 0]
    assert copy.read_bytes() == before

    # Exported too, Blood clots stay whole; a missing container is said plainly.
    naming["rules"]["local:Blood clots"] = {"subtract": True, "export": True}
    result = names.export(root, config, image, output, True, naming)
    assert shape({f["properties"]["name"]: f for f in result["geojson"]["features"]}["Blood clots"]["geometry"]).area == 100
    try:
        names.export(root, config, image, output, True, {**naming, "inside": "imported:stroma"})
        raise AssertionError("expected an error")
    except ValueError as error:
        assert "No stroma (imported)" in str(error)


def test_keep_inside_keeps_pieces_renamed_in_the_same_layer(tmp_path):
    """A tissue piece renamed by hand (lymph_node) is not cut away by Keep inside the tissue."""
    root, config, (image,), output = _project(tmp_path, "A_roi_1")
    tissue = annotation.valid_tissue(output, annotation.sample_id(image.name))
    edits.replace(tissue, [_feature(box(0, 0, 50, 50)), _feature(box(100, 100, 150, 150), name="lymph_node", label="lymph_node", named=True)],
                  edits.read(tissue)[1], "test", kind="tissue")
    naming = {**names.EXPORT_DEFAULTS, "inside": "local:tissue", "rules": {"local:tissue": {"export": True}, "local:lymph_node": {"export": True}}}
    assert names.export(root, config, image, output, True, naming)["names"] == ["lymph_node", "tissue"]


def test_files_match_the_image_they_name():
    images = ["/p/P11_roi_1.ome.tiff", "/p/P11_roi_10.ome.tiff", "/p/P11_roi_1_bg_subtracted.ome.tiff", "/p/P2.ome.tiff"]
    assert names.match_image(images, "P11_roi_1_old.geojson") == images[0]
    assert names.match_image(images, "P11-ROI1 tumour.geojson") == images[0]
    assert names.match_image(images, "old_P11_roi_10.geojson") == images[1]
    assert names.match_image(images, "P11_roi_1_bg_subtracted_v2.geojson") == images[2]
    assert names.match_image(images, "P12_roi_3.geojson") is None
    assert names.match_image(images, "P21.geojson") is None  # P2 inside P21 is not a match
    assert names.match_image(["/p/Sample_A12.ome.tiff", "/p/Other_B7.ome.tiff"], "SampleA12.geojson") == "/p/Sample_A12.ome.tiff"  # fuzzy


def test_export_rules_apply_to_every_ticked_image(tmp_path):
    """Selected: each image applies the same rules; a name it lacks is skipped; an image without the container (or
    without anything to export) is listed as not exported; no .sources.json is written."""
    from spacemap_gui.studio import tasks
    from spacemap_gui.studio.settings import read_config

    root, config, images, output = _project(tmp_path, "A_roi_1", "A_roi_2", "A_roi_3")
    _old_file(root, images[0], ("tissue", (0, 0, 60, 60)), ("tumor", (0, 0, 20, 20)))
    _old_file(root, images[1], ("tissue", (0, 0, 60, 60)))
    names.save_export_settings(root, {"inside": "imported:tissue", "rules": {"imported:tissue": {"export": True}, "imported:tumor": {"export": True}}})
    result = tasks.export_annotations({"root": str(root), "images": [str(i) for i in images], "merge": True}, None, lambda *a: None)
    assert [r["image"] for r in result["images"]] == ["A_roi_1.ome.tiff", "A_roi_2.ome.tiff"]
    assert result["failed"][0]["image"] == "A_roi_3.ome.tiff"
    config = read_config(root)
    folder = names.export_folder(root, names.export_settings(root, config))
    exported = lambda image: sorted(f["properties"]["name"] for f in json.loads((folder / (annotation.sample_id(image.name) + ".geojson")).read_text())["features"])
    assert exported(images[0]) == ["tissue", "tumor"] and exported(images[1]) == ["tissue"]
    assert not list(root.rglob("*.sources.json"))
    # Rules change one at a time; Layers shows them as ticks; a rename takes its rule along.
    names.save_export_settings(root, {"rules": {"imported:tumor": {"subtract": True}}})
    config = read_config(root)
    assert names.export_settings(root, config)["rules"]["imported:tumor"] == {"export": True, "subtract": True}
    assert names.choices(root, config, images[0], output)["A_roi_1_old::tumor::"]
    names.rename(root, config, images[0], output, "A_roi_1_old::tumor::", "Tumor")
    assert names.export_settings(root, read_config(root))["rules"]["imported:Tumor"]["export"]


def test_delete_results_sets_a_stage_aside(tmp_path):
    """Deleting Classification's results moves its polygons and run aside (the letter turns grey), keeps its settings
    and leaves the tissue; deleting the export removes the named copy too."""
    from spacemap_gui.studio import classification

    root = tmp_path / "study"
    root.mkdir()
    config = {"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}
    (root / "study.json").write_text(json.dumps(config))
    images_folder, output = annotation.folders(root, {})
    images_folder.mkdir(parents=True, exist_ok=True)
    image = _image(images_folder / "A_roi_1.ome.tiff")
    sample = annotation.sample_id(image.name)
    tissue = annotation.valid_tissue(output, sample)
    tissue.parent.mkdir(parents=True, exist_ok=True)
    edits.replace(tissue, [_feature(box(0, 0, 40, 40))], edits.revision(tissue), "tissue", kind="tissue")
    classes = annotation.classes(output, sample)
    edits.replace(classes, [_feature(box(0, 0, 10, 10), classification={"name": "Class 1"})], edits.revision(classes), "classes")
    runs = classification.directory(root, image)
    runs.mkdir(parents=True)
    (runs / "abc.npz").write_bytes(b"x")
    (runs / "settings.json").write_text(json.dumps({"options": {"k": 5}, "run_id": "abc"}))
    plan = lambda: {r["image"]: r for r in annotation.images_plan(root, {}, config)["rows"]}[image.name]
    assert plan()["done"]["classes"] and plan()["classified"]

    moved = names.delete_results(root, config, image, "classes")
    assert any(m.endswith("_abc.npz") for m in moved) and all("outputs/studio/deleted" in m for m in moved)
    assert not plan()["done"]["classes"] and not plan()["classified"] and plan()["done"]["tissue"]
    assert json.loads((runs / "settings.json").read_text()) == {"options": {"k": 5}}

    result = names.export(root, config, image, output, True, {**names.export_settings(root, config), "rules": {"local:tissue": {"export": True}}})
    assert plan()["done"]["export"]
    names.delete_results(root, config, image, "export")
    assert not plan()["done"]["export"] and not (tmp_path / result["saved_as"]).exists()
    assert names.delete_results(root, config, image, "zones") == []


def test_exports_are_not_read_back_and_left_out_names_are_said(tmp_path):
    """An exported file in the Output folder is not a source of the image's annotations; a ticked annotation the
    Keep inside leaves nothing of is listed in left_out; exports go to the Output folder whatever was typed before."""
    root, config, (image,), output = _project(tmp_path, "A_roi_1")
    _old_file(root, image, ("tumor", (0, 0, 20, 20)), ("epithelia", (30, 30, 40, 40)))
    naming = {**names.EXPORT_DEFAULTS, "folder": "/elsewhere", "inside": "imported:tumor",
              "rules": {"imported:tumor": {"export": True}, "imported:epithelia": {"export": True}}}
    result = names.export(root, config, image, output, True, naming)
    assert result["names"] == ["tumor"] and result["left_out"] == ["epithelia (imported)"]
    assert Path(result["saved_as"]).parent == annotation.main_output(root, {})
    found = names.groups(root, config, image, output)
    assert not any(Path(f).name == "A_roi_1.geojson" for g in found for f in g["files"])  # the export is not read back
    assert {g["rule"] for g in found} == {"local:tissue", "imported:tumor", "imported:epithelia"}


def test_small_pieces_of_keep_inside_are_dropped_with_their_contents(tmp_path):
    """Keep inside an imported tissue of a big piece and a small one; the small piece (and the tumor in it) goes, and so
    does a sliver the subtracted Blood clots cut off the big piece."""
    from shapely.geometry import shape
    from spacemap.viewer.sources import image_info

    root, config, (image,), output = _project(tmp_path, "A_roi_1")
    sample = annotation.sample_id(image.name)
    clots = artifact_learning.polygons_path(output, sample, "clot")
    edits.replace(clots, [_feature(box(0, 50, 60, 52))], edits.revision(clots), "clots")  # cuts a 60 x 8 strip off the big piece
    _old_file(root, image, ("tissue", (0, 0, 60, 60)), ("tissue", (100, 100, 110, 110)),
              ("tumor", (0, 0, 10, 10)), ("tumor", (101, 101, 104, 104)))
    microns = image_info(str(image)).pixel_size_um
    rules = {"imported:tissue": {"export": True}, "imported:tumor": {"export": True}, "local:Blood clots": {"subtract": True}}
    naming = {**names.EXPORT_DEFAULTS, "inside": "imported:tissue", "rules": rules, "smallest_um2": 600 * microns ** 2}
    result = names.export(root, config, image, output, True, naming)
    shapes = {f["properties"]["name"]: shape(f["geometry"]) for f in result["geojson"]["features"]}
    assert result["small_pieces_dropped"] == 2  # the 10 x 10 piece and the 60 x 8 strip
    assert shapes["tissue"].area == 60 * 50 and shapes["tumor"].area == 100
    naming["smallest_um2"] = 0
    assert names.export(root, config, image, output, True, naming)["small_pieces_dropped"] == 0


def test_a_folder_of_geojson_is_imported_listed_and_removed(tmp_path):
    """Import a folder: each file to the image it names, once; an export and an unknown name are not imported. The
    list shows the copies; removing one takes it off its image. Segmentation offers imported:tissue for every image."""
    from spacemap_gui.studio import segmentation

    root, config, images, output = _project(tmp_path, "A_roi_1", "A_roi_2")
    folder = tmp_path / "old"
    folder.mkdir()
    feature = lambda name, bounds: _feature(box(*bounds), classification={"name": name})
    (folder / "A-ROI1 old.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature("tissue", (0, 0, 30, 30))]}))
    (folder / "A_roi_2_old.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature("tissue", (0, 0, 20, 20))]}))
    (folder / "B_roi_9.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature("tissue", (0, 0, 20, 20))]}))
    result = names.import_directory(root, config, folder)
    assert sorted((d["file"], d["image"]) for d in result["imported"]) == [("A-ROI1 old.geojson", "A_roi_1.ome.tiff"), ("A_roi_2_old.geojson", "A_roi_2.ome.tiff")]
    assert result["unmatched"] == ["B_roi_9.geojson"]
    assert names.import_directory(root, config, folder)["count"] == 0  # already imported: skipped
    listed = names.imports(root, config)
    assert sorted(r["image"] for r in listed) == ["A_roi_1.ome.tiff", "A_roi_2.ome.tiff"] and all(r["names"] == ["tissue"] for r in listed)

    saved = segmentation.saved_annotations(root, config, {}, [i.name for i in images])
    assert all("imported:tissue" in saved[i.name] and "local:tissue" in saved[i.name] for i in images)
    groups = segmentation._area_groups(saved["A_roi_1.ome.tiff"])
    assert [g for g, _ in groups] == ["Local", "Imported"] and ["imported:tissue", "tissue"] in groups[1][1]
    assert segmentation._shown("tissue") == "local:tissue" and segmentation._shown("__whole__") == "__whole__"

    assert names.remove_imports(root, config, [{"image": listed[0]["image"], "path": listed[0]["path"]}]) == 1
    assert len(names.imports(root, config)) == 1
