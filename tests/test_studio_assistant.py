"""Requests stay deterministic when possible; model output remains checked data."""

import json

import pytest

from spacemap_gui.studio import assistant
from spacemap_gui.viewer import registry

CASES = [
    ("open P11 roi 2", "open_image", {"image": "P11 roi 2"}),
    ("load image P11_roi_2.ome.tiff", "open_image", {"image": "P11_roi_2.ome.tiff"}),
    ("show image P11 roi 2", "open_image", {"image": "P11 roi 2"}),
    ("show markers CD8 and CD3", "show_markers", {"markers": ["CD8", "CD3"]}),
    ("show CD8, CD3", "show_markers", {"markers": ["CD8", "CD3"]}),
    ("turn on CD8 & CD3 on P11 roi 2", "show_markers", {"markers": ["CD8", "CD3"], "image": "P11 roi 2"}),
    ("use marker set Immune", "apply_marker_set", {"set": "Immune"}),
    ("colour cells by phenotype_ml_consensus", "colour_by", {"column": "phenotype_ml_consensus"}),
    ("color by phenotype_ref1", "colour_by", {"column": "phenotype_ref1"}),
    ("show me the density of CD8 cells on P11_roi_2", "density", {"phenotype": "CD8", "image": "P11_roi_2"}),
    ("density of B cells in this image", "density", {"phenotype": "B"}),
    ("count CD8 cells", "count_cells", {"phenotype": "CD8", "scope": "image"}),
    ("how many cells are there in the project", "count_cells", {"scope": "project"}),
    ("count B cells across all ROIs", "count_cells", {"scope": "project", "phenotype": "B"}),
    ("zoom to cell 708926", "zoom_to_cell", {"cell": 708926}),
    ("set threshold for CD8 to 125", "set_threshold", {"marker": "CD8", "value": 125.0}),
    ("change CD8 threshold to 250 raw units", "set_threshold", {"marker": "CD8", "value": 250.0}),
    ("save thresholds", "save_thresholds", {}),
    ("save current view", "save_image", {}),
    (
        "show me markers CD8 and CD3 on image P11_roi_2",
        "show_markers",
        {"markers": ["CD8", "CD3"], "image": "P11_roi_2"},
    ),
]


@pytest.mark.parametrize("prefix", ["", "Please ", "Can you ", "Could you please ", "I would like you to "])
@pytest.mark.parametrize("text,name,args", CASES)
def test_one_hundred_common_phrasings(prefix, text, name, args):
    result = assistant.parse(prefix + text + "?")
    assert result == {"name": name, "args": args}


@pytest.fixture
def channel(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    registry.publish("assistant-test", {"id": "empty", "layers": []})
    monkeypatch.setattr(
        assistant,
        "viewer_context",
        lambda _: {"markers": ["CD8", "CD3"], "columns": ["phenotype_ref1"], "image": "P11_roi_2"},
    )
    return "assistant-test"


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(list(messages))
        return next(self.replies)


def test_common_requests_never_call_the_model(channel):
    client = Client([])
    assert assistant.propose(channel, "show CD8", client=client)["source"] == "instant"
    assert not client.messages


def test_model_output_gets_one_repair_and_has_no_paths_or_code(channel):
    client = Client(
        ['{"name":"run_python","args":{"code":"bad"}}', '{"name":"show_markers","args":{"markers":["CD8"]}}']
    )
    result = assistant.propose(channel, "I need a clearer look at the CD8 signal", client=client)
    assert result["command"]["name"] == "show_markers" and len(client.messages) == 2
    assert "Correct the JSON once." in client.messages[1][-1]["content"]


def test_model_clarifies_and_rejects_unknown_fields(channel):
    client = Client(['{"question":"Which phenotype should I count?"}'])
    assert assistant.propose(channel, "How abundant are they?", client=client)["status"] == "question"
    invalid = '{"name":"open_image","args":{"image":"test"},"confirmed":true}'
    client = Client([invalid, invalid])
    assert assistant.propose(channel, "Bring the interesting sample up", client=client)["status"] == "question"
    assert len(client.messages) == 2


@pytest.mark.parametrize("text", ["delete this project", "run shell commands", "open P11 then assign cells", "", None])
def test_unrecognised_is_not_an_action(text):
    assert assistant.parse(text) is None


def test_description_uses_only_submitted_preview_and_is_labelled(channel):
    import base64
    import io
    from PIL import Image

    target = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(target, format="PNG")
    preview = "data:image/png;base64," + base64.b64encode(target.getvalue()).decode()
    client = Client(["The view is mostly blue."])
    result = assistant.describe(channel, preview, client=client)
    assert result["status"] == "description" and result["message"].startswith("AI description:")
    assert client.messages[0][-1]["content"][1]["image_url"]["url"] == preview
    assert "Do not infer diagnoses" in client.messages[0][0]["content"]


@pytest.mark.parametrize("preview", ["https://outside/image.png", "data:image/png;base64,notbase64", None])
def test_description_rejects_invalid_or_remote_images(channel, preview):
    with pytest.raises(ValueError):
        assistant.describe(channel, preview, client=Client([]))


def test_selection_phrasings_stay_checked():
    from spacemap_gui.studio.assistant import parse

    assert parse("select cells where CD8 Raw > 100 and CD3 Arcsinh >= 2")["args"]["rule"] == {
        "conditions": [
            {"marker": "CD8", "transform": "Raw", "op": ">", "value": 100.0},
            {"marker": "CD3", "transform": "Arcsinh", "op": ">=", "value": 2.0},
        ],
        "join": "all",
        "scope": "image",
    }
    assert parse("select cells where CD8 > 100 across all ROIs")["args"]["rule"]["scope"] == "project"
    assert parse("assign selected cells as CD8 T in phenotype_ref1") == {
        "name": "assign_selection",
        "args": {"phenotype": "CD8 T", "column": "phenotype_ref1"},
    }
    assert parse("show a heatmap of the selection")["name"] == "selection_heatmap"
    assert parse("select cells where CD8 > 100 and CD3 > 20 or CD20 > 200") is None


def test_planned_plain_selection_and_assignment_examples():
    from spacemap_gui.studio.assistant import parse

    selected = parse("select cells with CD3e above 250 and CD8 Z1 above 1.5")
    assert selected["name"] == "select_cells" and selected["args"]["rule"]["conditions"][1]["value"] == 1.5
    assert parse("assign them as CD8 T") == {"name": "assign_selection", "args": {"phenotype": "CD8 T"}}
