"""Studio's Registration step runs Tab 1's own pipeline for each pair."""

from pathlib import Path

import pytest

from spacemap_gui.studio import registration
from test_studio_commands import run, setup  # noqa: F401  (fixture)


def test_pairs_need_both_images(tmp_path):
    for name in ["data/codex/P11_codex.qptiff", "data/he/P11_he.qptiff", "data/codex/P12_codex.qptiff"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    found = registration.pairs(tmp_path, {})
    assert list(found) == ["P11"] and found["P11"]["he"].endswith("P11_he.qptiff")
    assert registration.output_for(tmp_path / "outputs/registered", "P11").name == "P11_merged.ome.tiff"


def test_options_are_checked():
    assert registration.check({}) == {"anchor": None, "rotation": 0, "keypoints": 10000}
    for wrong in ({"rotation": 45}, {"keypoints": 10}, {"keypoints": True}, {"anchor": ""}):
        if wrong == {"anchor": ""}:
            assert registration.check(wrong)["anchor"] is None
            continue
        with pytest.raises(ValueError):
            registration.check(wrong)


def test_register_calls_tab1_in_order_and_keeps_done_pairs(tmp_path, monkeypatch):
    import spacemap.registration as tab1

    calls = []
    monkeypatch.setattr(registration, "pixel_size", lambda path, fallback: fallback)
    monkeypatch.setattr(
        tab1,
        "preprocess_he_image",
        lambda he, folder, mpp, pre_rotation_deg=0.0: calls.append(("he", mpp, pre_rotation_deg)) or "resized",
    )
    monkeypatch.setattr(
        tab1,
        "extract_dapi_channel",
        lambda codex, folder, reference_channel=None: calls.append(("anchor", reference_channel)) or "dapi",
    )
    monkeypatch.setattr(
        tab1,
        "register_images",
        lambda resized, dapi, folder, mpp, reference_channel=None, n_keypoints=0: (
            calls.append(("align", n_keypoints)) or "aligned"
        ),
    )

    def merge(paths, output, mpp, configs, provenance=None):
        calls.append(("merge", paths, mpp))
        assert provenance["CODEX"] == "c11" and provenance["Anchor"] == "DAPI" and provenance["Keypoints"] == "5000"
        Path(output).touch()

    monkeypatch.setattr(tab1, "merge_multi_round_ome_tiff", merge)
    done = tmp_path / "P10_merged.ome.tiff"
    done.touch()
    items = [
        {"sample": "P10", "codex": "c10", "he": "h10", "output": str(done)},
        {"sample": "P11", "codex": "c11", "he": "h11", "output": str(tmp_path / "P11_merged.ome.tiff")},
    ]
    written = registration.register(items, {"anchor": "DAPI", "rotation": 90, "keypoints": 5000}, lambda *_: None)
    assert written == [str(tmp_path / "P11_merged.ome.tiff")]
    assert [call[0] for call in calls] == ["he", "anchor", "align", "merge"]
    assert calls[0] == ("he", {"hne": registration.HE_MPP, "codex": registration.CODEX_MPP}, 90.0)
    assert calls[1] == ("anchor", "DAPI") and calls[2] == ("align", 5000)
    assert calls[3] == ("merge", ["c11", "aligned"], registration.CODEX_MPP)
    assert (tmp_path / registration.INTERMEDIATE).is_dir()


def test_register_command_finds_the_pair_and_starts_a_job(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs

    root = setup[2]
    for name in ["data/codex/P11_roi_2_codex.qptiff", "data/he/P11_roi_2_he.qptiff"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
    started = {}
    monkeypatch.setattr(
        jobs,
        "start",
        lambda project, kind, settings, request_id=None: started.update(kind=kind, settings=settings) or {"id": "x"},
    )
    result = run(setup, "register_pairs", {"rotation": 180, "keypoints": 2000}, confirmed=True)
    assert result["status"] == "done", result
    pair = started["settings"]["pairs"][0]
    assert started["kind"] == "register_pairs" and pair["sample"] == "P11_roi_2"
    assert pair["output"].endswith("outputs/registered/P11_roi_2_merged.ome.tiff")
    assert started["settings"]["options"] == {"anchor": None, "rotation": 180, "keypoints": 2000}
    wrong = run(setup, "register_pairs", {"rotation": 45}, confirmed=True)
    assert wrong["status"] == "error"


def _project(root, codex, he, config=None):
    import json

    for folder, names in (("data/codex", codex), ("data/he", he)):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / folder / name).touch()
    (root / f"{root.name}.json").write_text(json.dumps({"project_name": root.name, "directory_mappings": {}, "tab_configurations": config or {}}))


def test_pairs_follow_tab1s_naming_rule_and_choices(tmp_path):
    from spacemap_gui.studio.settings import read_config

    _project(tmp_path, ["CODEX_CRC02.qptiff", "CODEX_CRC03.qptiff", "slide7_codex.qptiff"], ["CRC02-HE.tif", "CRC03-HE.tif", "S7 HE scan.tif"])
    found = registration.plan(tmp_path, {}, read_config(tmp_path))
    assert [(row["sample"], row["he"], row["follows_rule"]) for row in found["rows"]] == [
        ("CODEX_CRC02", None, False), ("CODEX_CRC03", None, False), ("slide7", None, True)]  # every image listed, none paired yet
    assert found["he_files"] == ["CRC02-HE.tif", "CRC03-HE.tif", "S7 HE scan.tif"] and found["ignored_codex"] == found["ignored_he"] == []
    found = registration.save_pairing(tmp_path, {"tab1_batch_codex_prefix": "CODEX_", "tab1_batch_codex_suffix": "", "tab1_batch_he_suffix": "-HE"},
                                      registration.pairing_revision(found))
    assert [(row["sample"], row["he"]) for row in found["rows"]] == [("CRC02", "CRC02-HE.tif"), ("CRC03", "CRC03-HE.tif"), ("slide7_codex", None)]
    found = registration.save_pairing(tmp_path, {"tab1_batch_codex_prefix": "", "tab1_batch_he_suffix": ""}, registration.pairing_revision(found))
    found = registration.save_pairing(
        tmp_path,
        {"choices": {"slide7_codex.qptiff": "S7 HE scan.tif", "CODEX_CRC02.qptiff": "CRC02-HE.tif", "CODEX_CRC03.qptiff": "<Skip>"}},
        registration.pairing_revision(found),
    )
    assert {row["sample"]: row["he"] for row in found["rows"]} == {"CODEX_CRC02": "CRC02-HE.tif", "CODEX_CRC03": None, "slide7_codex": "S7 HE scan.tif"}
    assert found["unpaired_he"] == ["CRC03-HE.tif"]
    assert sorted(registration.pairs(tmp_path, {})) == ["CODEX_CRC02", "slide7_codex"]
    saved = read_config(tmp_path)["tab_configurations"]["tab1_registration"]  # where classic Tab 1 keeps them
    assert saved["tab1_pair_override_slide7_codex.qptiff"] == "S7 HE scan.tif" and saved["tab1_batch_he_suffix"] == ""


def test_pairing_changes_are_checked(tmp_path):
    from spacemap_gui.studio.settings import read_config

    _project(tmp_path, ["P1_codex.qptiff"], ["P1_he.tif"])
    (tmp_path / "other/images").mkdir(parents=True)
    (tmp_path / "other/images/P1_he.tif").touch()
    found = registration.plan(tmp_path, {}, read_config(tmp_path))
    revision = registration.pairing_revision(found)
    assert registration.image_folders(tmp_path) == ["data/codex", "data/he", "other/images"]
    for wrong in (
        {"folders": {"codex": "../elsewhere"}},
        {"folders": {"he": "/data/he"}},
        {"folders": {"he": "missing/folder"}},  # not there
        {"folders": {"registered": "no/parent/here"}},
        {"choices": {"P1_codex.qptiff": "missing.tif"}},
        {"choices": {"nope.qptiff": "<Skip>"}},
        {"tab1_batch_he_suffix": "a/b"},
        {"colour": "x"},
    ):
        with pytest.raises(ValueError):
            registration.save_pairing(tmp_path, wrong, revision)
    with pytest.raises(ValueError, match="another window"):
        registration.save_pairing(tmp_path, {"tab1_batch_he_suffix": "_he"}, "stale")
    (tmp_path / "outputs").mkdir()
    elsewhere = tmp_path.parent / (tmp_path.name + "_scans")  # a folder outside the project, typed as a full path
    elsewhere.mkdir()
    found = registration.save_pairing(tmp_path, {"folders": {"he": str(elsewhere), "registered": "outputs/aligned"}}, revision)
    assert found["folders"] == {"codex": str(tmp_path / "data/codex"), "he": str(elsewhere), "registered": str(tmp_path / "outputs/aligned")}
    assert read_config(tmp_path)["directory_mappings"]["batch_he_dir"] == str(elsewhere)
    assert read_config(tmp_path)["directory_mappings"]["registration_output_dir"] == "${PROJECT_ROOT}/outputs/aligned"


def test_registering_again_keeps_the_earlier_result(tmp_path, monkeypatch):
    output = tmp_path / "P1_merged.ome.tiff"
    output.write_text("first")
    item = {"sample": "P1", "codex": "c", "he": "h", "output": str(output), "again": True}

    def fails(*args):
        output.write_text("partial")
        raise RuntimeError("palom failed")

    monkeypatch.setattr(registration, "_register_one", fails)
    with pytest.raises(RuntimeError):
        registration.register([item], {}, lambda *_: None)
    assert output.read_text() == "first" and not list((tmp_path / "previous").iterdir())  # put back
    monkeypatch.setattr(registration, "_register_one", lambda item, *args: output.write_text("second") or str(output))
    registration.register([item], {}, lambda *_: None)
    registration.register([item], {}, lambda *_: None)
    assert output.read_text() == "second"
    assert sorted(path.name for path in (tmp_path / "previous").iterdir()) == ["P1_merged.ome.tiff", "P1_merged_2.ome.tiff"]
    assert (tmp_path / "previous/P1_merged.ome.tiff").read_text() == "first"


def test_the_matched_points_figure_is_found_by_the_he_name(tmp_path):
    from spacemap_gui.studio.settings import read_config

    _project(tmp_path, ["P1_codex.qptiff"], ["P1_he.tif"])
    config = read_config(tmp_path)
    assert registration.feature_matches(tmp_path, {}, config, "P1") is None
    folder = tmp_path / "outputs/registered" / registration.INTERMEDIATE
    folder.mkdir(parents=True)
    (folder / "P1_he_registered_feature_matches.png").touch()
    assert registration.feature_matches(tmp_path, {}, config, "P1").name == "P1_he_registered_feature_matches.png"
    with pytest.raises(ValueError):
        registration.feature_matches(tmp_path, {}, config, "P9")


def test_register_command_takes_picked_pairs(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs

    root = setup[2]
    for name in ["data/codex/A_codex.qptiff", "data/he/A_he.tif", "data/codex/B_codex.qptiff", "data/he/B_he.tif"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
    (root / "outputs/registered").mkdir(parents=True, exist_ok=True)
    (root / "outputs/registered/A_merged.ome.tiff").touch()
    started = []
    monkeypatch.setattr(jobs, "start", lambda project, kind, settings, request_id=None: started.append(settings) or {"id": "x"})
    result = run(setup, "register_pairs", {"scope": "pairs", "samples": ["A", "B"]}, confirmed=True)
    assert result["status"] == "done" and [item["sample"] for item in started[-1]["pairs"]] == ["B"]
    result = run(setup, "register_pairs", {"scope": "pairs", "samples": ["A"], "again": True}, confirmed=True)
    assert result["status"] == "done" and started[-1]["pairs"][0]["again"] is True and "previous/" in result["message"]
    assert run(setup, "register_pairs", {"scope": "pairs", "samples": ["A"]}, confirmed=True)["status"] == "error"
    assert run(setup, "register_pairs", {"scope": "pairs", "samples": ["Z"]}, confirmed=True)["status"] == "error"


def test_each_pair_has_its_own_layers_and_extra_codex_images(tmp_path, monkeypatch):
    import spacemap.background_subtraction_utils as bsu
    from spacemap_gui.studio.settings import read_config

    _project(tmp_path, ["P1_codex.qptiff", "P2_codex.qptiff"], ["P1_he.tif", "P2_he.tif"])
    (tmp_path / "data/codex2").mkdir()
    for name in ("P1_codex.qptiff", "other_run.qptiff"):
        (tmp_path / "data/codex2" / name).touch()
    monkeypatch.setattr(bsu, "get_channel_info", lambda path: (["CD3", "DAPI", "CD8"], [(255, 0, 0), (0, 0, 255), (0, 255, 0)], [{}] * 3, {}))
    config = read_config(tmp_path)
    found = registration.pair_markers(tmp_path, {}, config, "P1")
    assert [image["label"] for image in found["images"]] == ["CODEX", "H&E"] and found["images"][0]["anchor"] == "DAPI"
    assert [c["name"] for c in found["images"][1]["channels"]] == ["HE_Red", "HE_Green", "HE_Blue"] and not found["edited"]
    assert found["mpp"]["he"] == {"detected": None, "fallback": registration.HE_MPP, "saved": None, "used": registration.HE_MPP}  # empty test files
    # An extra CODEX folder: matched by name, or picked by hand.
    plan = registration.plan(tmp_path, {}, config)
    plan = registration.save_pairing(tmp_path, {"extra_folders": ["data/codex2"]}, registration.pairing_revision(plan))
    rows = {row["sample"]: row for row in plan["rows"]}
    assert rows["P1"]["extra"][0]["file"] == "P1_codex.qptiff" and rows["P2"]["extra"][0]["file"] is None
    plan = registration.save_pairing(tmp_path, {"extra_choices": {"P2_codex.qptiff": ["other_run.qptiff"]}}, registration.pairing_revision(plan))
    assert {row["sample"]: row for row in plan["rows"]}["P2"]["extra"][0]["file"] == "other_run.qptiff"
    config = read_config(tmp_path)
    found = registration.pair_markers(tmp_path, {}, config, "P1")
    assert [image["label"] for image in found["images"]] == ["CODEX (1)", "CODEX (2)", "H&E"]
    saved = registration.save_pair_markers(tmp_path, "P1", {
        "images": [{"keep": [0, 1, 2], "rename": {"2": "CD8a"}, "colors": {"0": "#ffffff"}, "anchor": "DAPI"},
                   {"keep": [0, 2], "anchor": "DAPI"}],  # the second DAPI is left out but still aligns
        "he": {"keep": [0, 1, 2]}, "mpp": {"he": 0.25}})
    assert saved["mpp"]["he"]["used"] == 0.25 and saved["mpp"]["he"]["saved"] == 0.25
    assert saved["edited"] and saved["images"][0]["channels"][2]["rename"] == "CD8a" and not saved["images"][1]["channels"][1]["keep"]
    assert {row["sample"]: row for row in registration.plan(tmp_path, {}, read_config(tmp_path))["rows"]}["P1"]["edited"]
    pairs = registration.pairs(tmp_path, {})
    assert pairs["P1"]["extra"][0].endswith("data/codex2/P1_codex.qptiff") and pairs["P1"]["markers"]["images"][1]["keep"] == [0, 2]
    for wrong in ({"images": [{"keep": [0], "anchor": "DAPI"}], "mpp": {"he": 0}}, {"images": [{"keep": []}]}, {"images": [{"keep": [0], "colors": {"0": "red"}, "anchor": "DAPI"}]}, {"other": 1}):
        with pytest.raises(ValueError):
            registration.save_pair_markers(tmp_path, "P1", wrong)


def test_extra_codex_images_are_aligned_to_the_first_and_merged(tmp_path, monkeypatch):
    import spacemap.registration as tab1

    calls = []
    monkeypatch.setattr(registration, "pixel_size", lambda path, fallback: fallback)
    monkeypatch.setattr(tab1, "preprocess_he_image", lambda he, folder, mpp, pre_rotation_deg=0.0: calls.append(("he", mpp)) or "resized")
    monkeypatch.setattr(tab1, "extract_dapi_channel", lambda codex, folder, reference_channel=None: calls.append(("anchor", codex, reference_channel)) or codex + ".anchor")
    monkeypatch.setattr(tab1, "register_images", lambda *args, **kwargs: "he_registered")

    def align(fixed, moving, codex, output, mpp, **kwargs):
        calls.append(("align", fixed, moving, kwargs["fixed_reference_channel"], kwargs["moving_reference_channel"]))
        Path(output).touch()

    def merge(paths, output, mpp, configs, provenance=None):
        calls.append(("merge", [Path(p).name for p in paths], configs, provenance["CODEX (2)"]))
        Path(output).touch()

    monkeypatch.setattr(tab1, "register_codex_to_codex", align)
    monkeypatch.setattr(tab1, "merge_multi_round_ome_tiff", merge)
    item = {"sample": "P1", "codex": "c1", "he": "h1", "extra": ["c2"], "output": str(tmp_path / "P1_merged.ome.tiff"),
            "markers": {"images": [{"keep": [0, 1], "rename": {"1": "N"}, "colors": {}, "anchor": "DAPI"}, {"keep": [2], "anchor": "Hoechst"}],
                        "he": {"keep": [0, 1, 2]}, "mpp": {"he": 0.125}}}
    registration.register([item], {"anchor": None, "rotation": 0, "keypoints": 1000}, lambda *_: None)
    assert ("anchor", "c1", "DAPI") in calls and ("anchor", "c2", "Hoechst") in calls
    assert ("align", "c1.anchor", "c2.anchor", "DAPI", "Hoechst") in calls
    merged = next(call for call in calls if call[0] == "merge")
    assert merged[1] == ["c1", "P1_codex2_registered_to_codex1.ome.tiff", "he_registered"]
    assert merged[2][0] == {"keep_indices": [0, 1], "rename_by_index": {1: "N"}, "colors_by_index": {}}
    assert merged[2][1]["keep_indices"] == [2] and "Hoechst" in merged[3]
    assert ("he", {"hne": 0.125, "codex": registration.CODEX_MPP}) in calls  # the pair's typed H&E pixel size


def test_a_pair_opens_with_its_he_over_its_codex_to_align(tmp_path, monkeypatch):
    import json

    import numpy as np
    import tifffile

    from spacemap_gui.studio import commands
    from spacemap_gui.viewer import registry

    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    root = tmp_path / "study"
    for folder in ("data/codex", "data/he"):
        (root / folder).mkdir(parents=True)
    tifffile.imwrite(root / "data/codex/P1_codex.ome.tif", np.zeros((2, 64, 64), np.uint16), metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}})
    tifffile.imwrite(root / "data/he/P1_he.tif", np.zeros((128, 128, 3), np.uint8), photometric="rgb")
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    registry.publish("pair-test", {"id": "empty", "layers": []}, project={"root": str(root), "mappings": {}})
    registration.save_pair_rotation(root, "P1", 12.5)
    spec, _, _, _, text = commands._compare_pair("pair-test", {"sample": "P1"}, {"kind": "none"})
    image = next(layer for layer in spec["layers"] if layer["type"] == "image" and layer["id"] != "he")
    he = next(layer for layer in spec["layers"] if layer["id"] == "he")
    assert registry.resolve(image["token"]).endswith("P1_codex.ome.tif") and registry.resolve(he["token"]).endswith("P1_he.tif")
    assert he["align"]["rotation"] == 12.5 and he["align"]["he_mpp"] == registration.HE_MPP and he["align"]["scaled"]
    assert "P1" in text
    saved = registration.save_pair_markers(root, "P1", {"images": [{"keep": [0, 1], "anchor": "DAPI"}]})
    assert registration.pairs(root, {})["P1"]["markers"]["he_rotation"] == 12.5 and saved["edited"]  # the turn is kept with the layers
    with pytest.raises(ValueError):
        registration.save_pair_rotation(root, "P1", 999)


def test_quick_align_places_the_he_where_it_belongs(tmp_path):
    import cv2
    import numpy as np
    import tifffile

    from spacemap_gui.studio.settings import read_config

    root = tmp_path
    _project(root, [], [])
    rng = np.random.default_rng(3)
    codex = np.zeros((1200, 1200), np.float32)  # nuclei-like spots on a tissue-shaped area, at 0.5 µm/px
    for x, y in rng.integers(150, 1050, (900, 2)):
        cv2.circle(codex, (int(x), int(y)), int(rng.integers(4, 9)), float(rng.uniform(0.5, 1)), -1)
    codex = cv2.GaussianBlur(codex, (0, 0), 1.5)
    tifffile.imwrite(root / "data/codex/P1_codex.ome.tif", (np.stack([codex, codex * 0.3]) * 4000).astype(np.uint16),
                     metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "Channel": {"Name": ["DAPI", "CD3"]}})
    # The H&E: the same tissue turned 25° and shifted (in CODEX pixels), dark nuclei on light, at 0.25 µm/px.
    angle = np.radians(25)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    true = np.hstack([turn, [[180.0], [-90.0]]])  # H&E (in CODEX pixels) -> CODEX
    he_small = cv2.warpAffine(codex, true, (1100, 1100), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    he = cv2.resize(he_small, (2200, 2200), interpolation=cv2.INTER_LINEAR)
    rgb = np.stack([255 - he * 200, 255 - he * 230, 255 - he * 120], axis=-1).clip(0, 255).astype(np.uint8)
    tifffile.imwrite(root / "data/he/P1_he.ome.tif", rgb, photometric="rgb", metadata={"axes": "YXS", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25})
    config = read_config(root)
    found = registration.quick_align(root, {}, config, "P1")
    assert found["pixel_sizes_ok"] and abs(found["zoom"] - 1) < 0.05 and abs(found["rotation"] - 25) < 2, found
    # Where the viewer puts the H&E's corners (its box turned about its centre) against where they belong.
    size = np.array([1100.0, 1100.0]) * found["zoom"]
    centre = np.array([found["x"], found["y"]]) + size / 2
    a = np.radians(found["rotation"])
    placed = lambda u: centre + np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ (np.asarray(u) * found["zoom"] - size / 2)  # noqa: E731
    for corner in ([0, 0], [1100, 0], [0, 1100], [1100, 1100]):
        assert np.linalg.norm(placed(corner) - true @ np.array([*corner, 1.0])) < 25, (corner, placed(corner), true @ np.array([*corner, 1.0]))
