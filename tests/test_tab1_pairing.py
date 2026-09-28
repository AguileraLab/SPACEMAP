"""Tab 1's pairs: the naming rule pairs automatically; every image stays pairable by hand."""

from spacemap_gui.tabs.tab1_registration import find_pairs

RULE = ("", "_codex", "", "_he")


def _pairs(found):
    return {row["sample"]: (row["codex"], row["he"]) for row in found["rows"] if row["he"]}


def test_names_that_follow_the_rule_pair_as_before():
    codex = ["P11_codex.qptiff", "P15_P16_codex.qptiff", "P28_codex.qptiff"]
    he = ["P11_he.qptiff", "P15_P16_he.qptiff", "P28_he.qptiff"]
    found = find_pairs(codex, he, *RULE, {})
    assert _pairs(found) == {"P11": ("P11_codex.qptiff", "P11_he.qptiff"), "P15_P16": ("P15_P16_codex.qptiff", "P15_P16_he.qptiff"),
                             "P28": ("P28_codex.qptiff", "P28_he.qptiff")}
    assert found["unpaired_he"] == [] and all(row["follows_rule"] for row in found["rows"])


def test_any_image_can_be_paired_by_hand():
    codex = ["P1_codex.qptiff", "slide7.qptiff"]
    he = ["P1_he.tif", "S7 HE scan.tif"]
    found = find_pairs(codex, he, *RULE, {})
    rows = {row["codex"]: row for row in found["rows"]}
    assert rows["slide7.qptiff"]["he"] is None and not rows["slide7.qptiff"]["follows_rule"]
    assert "S7 HE scan.tif" in found["he_files"] and found["unpaired_he"] == ["S7 HE scan.tif"]
    found = find_pairs(codex, he, *RULE, {"slide7.qptiff": "S7 HE scan.tif", "P1_codex.qptiff": "<Skip>"})
    assert _pairs(found) == {"slide7": ("slide7.qptiff", "S7 HE scan.tif")} and found["unpaired_he"] == ["P1_he.tif"]
    assert find_pairs(codex, he, *RULE, {"P1_codex.qptiff": "gone.tif"})["rows"][0]["choice"] == "<Auto>"


def test_two_codex_images_with_one_sample_name_are_not_paired():
    found = find_pairs(["P1_codex.qptiff", "P1.qptiff"], ["P1_he.tif"], *RULE, {"P1.qptiff": "P1_he.tif"})
    assert all(row["duplicate"] and row["he"] is None for row in found["rows"]) and _pairs(found) == {}


def test_one_folder_for_both():
    names = ["P1_codex.qptiff", "P1_he.tif", "P2.tif"]
    found = find_pairs(names, names, *RULE, {}, same_folder=True)
    assert [row["codex"] for row in found["rows"]] == ["P1_codex.qptiff", "P2.tif"]
    assert found["he_files"] == ["P1_he.tif", "P2.tif"] and _pairs(found) == {"P1": ("P1_codex.qptiff", "P1_he.tif")}
