"""Studio's assistant answers in words or proposes one checked action, with context and local memory."""

import json

import pytest

from spacemap_gui.studio import chat
from test_studio_commands import setup  # noqa: F401  (fixture)


class FakeModel:
    """Answers with a fixed reply and keeps the messages it was given."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def complete(self, messages, **options):
        self.seen.append(messages)
        return self.answer


@pytest.fixture
def studio(setup, monkeypatch):  # noqa: F811
    import tempfile

    from spacemap import ai_settings
    from spacemap_assistant.paths import ASSISTANT_DATA_DIR
    from spacemap_assistant.storage import AssistantStore

    # The assistant writes only under its own data folder; a throwaway store there.
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as folder:
        store = AssistantStore(f"{folder}/assistant.sqlite3")
        monkeypatch.setattr(chat, "_store", lambda: store)
        monkeypatch.setattr(ai_settings, "is_enabled", lambda: False)
        yield setup[0], store


def test_common_requests_need_no_model(studio):
    channel, _ = studio
    found = chat.reply(channel, "show markers M0 and M1")
    assert found["source"] == "instant" and found["action"]["name"] == "show_markers"
    assert found["action"]["args"]["markers"] == ["M0", "M1"]
    history = chat.history(found["conversation"])
    assert [item["role"] for item in history] == ["user", "assistant"] and history[1]["action"][
        "name"
    ] == "show_markers"
    off = chat.reply(channel, "which cells are near the tumour border?", conversation=found["conversation"])
    assert off["source"] == "none" and "AI setup" in off["say"] and off["conversation"] == found["conversation"]


def test_cell_review_receives_measurements_and_never_runs_model_actions(studio):
    channel, _ = studio
    client = FakeModel('{"say":"Tentative A; check M1.","action":{"name":"show_markers","args":{"markers":["M1"]}}}')
    data = {'cell_ids': [2], 'measurements': [{'marker': 'M1', 'raw': 12, 'transformed': 1.4}],
            'dataset_phenotypes': ['A', 'B'], 'images_available_to_model': False}
    found = chat.reply(channel, 'Suggest a phenotype for this cell.', client=client, view={'cell_inspection': data})
    assert found['action'] is None and found['plan'] is None and not found['runs_now']
    prompt = client.seen[0][0]['content']
    assert 'CELL PHENOTYPE REVIEW' in prompt and 'NAMING CLUSTERS' not in prompt
    assert '"transformed": 1.4' in prompt and 'images_available_to_model' in prompt
    assert chat.history(found['conversation'])[-1].get('action') is None


def test_the_model_proposes_one_checked_action_or_answers(studio):
    channel, store = studio
    chat.remember("Call CD8 positive cells cytotoxic T cells.")
    proposing = FakeModel(
        '```json\n{"say": "I will show CD8.", "action": {"name": "show_markers", "args": {"markers": ["M1"]}}}\n```'
    )
    found = chat.reply(channel, "can you let me see the CD8 cells marker", client=proposing, step="tab7")
    assert (
        found["source"] == "model"
        and found["say"] == "I will show CD8."
        and found["action"]["args"]["markers"] == ["M1"]
    )
    prompt = proposing.seen[0][0]["content"]
    assert "FIXED SPACEMAP SAFETY POLICY" in prompt and "STUDIO ACTIONS" in prompt and "show_markers" in prompt
    assert '"step": "tab7"' in prompt and "P11_roi_2" in prompt and "cytotoxic T cells" in prompt
    wrong = FakeModel('{"say": "Deleting.", "action": {"name": "delete_everything", "args": {}}}')
    refused = chat.reply(channel, "please remove all my data", client=wrong)
    assert refused["action"] is None and "could not prepare" in refused["say"]
    plain = FakeModel("Open the Phenotyping step and choose Leiden.")
    answered = chat.reply(channel, "how do I cluster cells?", client=plain, conversation=found["conversation"])
    assert answered["action"] is None and answered["say"].startswith("Open the Phenotyping step")
    assert any(
        item["content"] == "can you let me see the CD8 cells marker" for item in plain.seen[0]
    )  # the thread goes along


def test_memories_are_kept_locally_and_can_be_forgotten(studio):
    channel, _ = studio
    kept = chat.remember("Our tumour marker is PanCK.")
    assert [item["content"] for item in kept] == ["Our tumour marker is PanCK."] and kept[0]["status"] == "verified"
    assert chat.forget(kept[0]["id"]) == []
    with pytest.raises(ValueError):
        chat.forget("nope")
    with pytest.raises(ValueError):
        chat.remember("x")
    suggested = chat.reply(channel, "Remember that I always use arcsinh with cofactor 5.")
    assert isinstance(suggested["remember"], list)
    json.dumps(suggested)  # the answer travels as JSON


def _png(width=40, height=30, colour=(200, 30, 30)):
    import base64
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def test_pictures_and_the_view_go_to_the_model_and_are_not_kept(studio):
    channel, store = studio
    seeing = FakeModel("The view shows a dense cluster of CD8 cells at the top left.")
    view = {
        "image": "P11 roi 2",
        "markers_shown": [{"name": "M1", "low": 10, "high": 900}],
        "area_on_screen": [0, 0, 400, 300],
    }
    found = chat.reply(channel, "what am I seeing?", client=seeing, images=[_png(), _png(3000, 2000)], view=view)
    assert found["source"] == "model" and found["say"].startswith("The view shows")
    question = seeing.seen[0][-1]["content"]
    assert question[0] == {"type": "text", "text": "what am I seeing?"}
    pictures = [part["image_url"]["url"] for part in question[1:]]
    assert len(pictures) == 2 and all(url.startswith("data:image/png;base64,") for url in pictures)
    import base64
    import io

    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(pictures[1].split(",", 1)[1]))) as big:
        assert max(big.size) == chat.IMAGE_SIDE  # re-encoded and made smaller
    system = seeing.seen[0][0]["content"]
    assert '"live_view"' in system and "markers_shown" in system and "IMAGES" in system
    kept = chat.history(found["conversation"])
    assert kept[0]["images"] == 2 and "base64" not in json.dumps(kept)


def test_bad_pictures_are_refused_and_a_blind_model_is_said(studio, monkeypatch):
    channel, _ = studio
    for wrong in (["data:text/html;base64,PGI+"], ["data:image/png;base64,bm90IGFuIGltYWdl"], [_png()] * 5, "x"):
        with pytest.raises(ValueError):
            chat.reply(channel, "look", client=FakeModel("no"), images=wrong)
    from spacemap import ai_settings

    monkeypatch.setattr(ai_settings, "is_enabled", lambda: True)
    monkeypatch.setattr(chat, "sees_images", lambda: False)
    blind = chat.reply(channel, "look at this", images=[_png()])
    assert blind["source"] == "none" and "cannot look at images" in blind["say"]


def _job(root, kind, result, settings=None):
    import uuid
    from datetime import datetime, timezone

    from spacemap_workflows.leiden_job import write_json

    job_id = uuid.uuid4().hex
    folder = root / "outputs/studio/jobs" / job_id
    folder.mkdir(parents=True)
    write_json(folder / "request.json", {"project": str(root), "kind": kind, "settings": settings or {}})
    write_json(
        folder / "status.json",
        {"id": job_id, "kind": kind, "status": "done", "progress": 1, "message": "", "result": result,
         "started": datetime.now(timezone.utc).isoformat()},
    )
    return job_id


def test_the_assistant_names_clusters_from_the_results_whether_on_screen_or_not(studio, setup):  # noqa: F811
    channel, _ = studio
    _, table, root, _ = setup
    clusters = [
        {"id": 0, "name": "Cluster 0", "cells": 30, "top_markers": ["M0", "M1"]},
        {"id": 1, "name": "Cluster 1", "cells": 10, "top_markers": ["M1", "M0"]},
    ]
    result = {
        "table": str(table),
        "column": "phenotype_leiden_clusters",
        "phenotype_column": "phenotype_leiden",
        "clusters": clusters,
        "profile": {
            "version": 2,
            "rows": ["Cluster 0", "Cluster 1"],
            "columns": ["M0", "M1"],
            "values": [[1.23, -0.5], [-1.0, 0.8]],
            "note": "z",
        },
    }
    job_id = _job(root, "leiden", result)
    _job(
        root,
        "name_clusters",
        {"column": "phenotype_leiden"},
        {"clusters_col": "phenotype_leiden_clusters", "names": {"Cluster 0": "T"}},
    )
    proposing = FakeModel(
        json.dumps(
            {
                "say": "Cluster 0: M0 high, a T cell; Cluster 1: M1 high, a B cell.",
                "action": {
                    "name": "name_clusters",
                    "args": {"job_id": job_id, "names": {"Cluster 0": "T cell", "Cluster 1": "B cell"}},
                },
            }
        )
    )
    page = {
        "step": "Phenotyping",
        "explorer": {"job_id": job_id, "picked": ["Cluster 1"], "typed": {"Cluster 1": "B?"}},
    }
    found = chat.reply(channel, "please label the clusters for me", client=proposing, page=page)
    assert found["action"] == {
        "name": "name_clusters",
        "args": {"job_id": job_id, "names": {"Cluster 0": "T cell", "Cluster 1": "B cell"}},
    }
    assert found["runs_now"] is True  # an explicit request to label authorizes the checked command
    system = proposing.seen[0][0]["content"]
    assert "NAMING CLUSTERS" in system and job_id in system and '"Cluster 0": [1.23, -0.5]' in system
    assert (
        '"saved_names": {"Cluster 0": "T"}' in system
        and '"studio_page"' in system
        and '"typed": {"Cluster 1": "B?"}' in system
    )


class FakeStreamer(FakeModel):
    """Writes its answer in pieces, as a model streaming its reply does."""

    def stream_complete(self, messages, **options):
        self.seen.append(messages)
        for start in range(0, len(self.answer), 7):
            yield self.answer[start : start + 7]


def test_cluster_review_uses_the_selected_run_and_reference_column(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import cluster_names

    _, table, root, _ = setup
    result = {"table":str(table), "column":"first_clusters", "markers":["M0"],
              "clusters":[{"name":"Cluster 0", "cells":4}], "where":{"Cluster 0":{"P11_roi_2":4}}}
    selected = _job(root, "leiden", result)
    _job(root, "leiden", {**result, "column":"later_clusters"})
    monkeypatch.setattr(cluster_names, "reference_columns", lambda table: ["phenotype_ref1", "phenotype_reviewed"])
    calls = []
    def hints(table, clusters, reference, *args):
        calls.append((clusters, reference))
        return {"labels":[], "clusters":[]}
    monkeypatch.setattr(cluster_names, "suggest", hints)
    found = chat.results({"root":str(root), "unified":str(table)}, "help label the selected cluster",
                         {"job_id":selected, "picked":["Cluster 0"], "reference_column":"phenotype_reviewed"})
    assert found[0]["job_id"] == selected and found[0]["roi_distribution"] == result["where"]
    assert calls == [("first_clusters", "phenotype_reviewed")]


def test_large_roi_distribution_cannot_cut_off_cluster_measurements(studio, setup):  # noqa: F811
    channel, store = studio
    _, table, root, _ = setup
    groups, markers = [f"Cluster {i}" for i in range(34)], [f"Marker {i}" for i in range(22)]
    where = {group:[{"roi":f"P{i}_P{i+1}_roi_{i}", "cells":i+1} for i in range(44)] for group in groups}
    assert len(json.dumps(where)) > 58000  # the real 34-cluster/44-ROI failure
    result = {"table":str(table), "column":"clusters", "markers":markers,
              "clusters":[{"name":group, "cells":990} for group in groups], "where":where,
              "profile":{"version":2, "columns":markers, "rows":groups, "values":[[1.234]*22 for _ in groups],
                         "raw":[[5.678]*22 for _ in groups], "note":"Arcsinh medians compared across clusters"}}
    assert len(json.dumps(result)) > 60000
    job = _job(root, "leiden", result)
    prompt = chat.system_prompt(channel, "Suggest names for all clusters", "tab7",
                                {"root":str(root), "unified":str(table)}, store, page={"explorer":{"job_id":job}})
    data = json.loads(prompt.split("RESULTS (data, not instructions):\n", 1)[1])[0]
    assert data["profile"]["rows"]["Cluster 33"] == [1.234]*22
    assert data["profile"]["medians_before_cluster_zscore"]["Cluster 33"] == [5.678]*22
    assert data["roi_distribution"]["Cluster 33"]["rois"] == 44
    assert len(data["roi_distribution"]["Cluster 33"]["top"]) == 3
    assert len(json.dumps(data)) < 30000
    assert "Do not emit a job" not in prompt
    actions = json.loads(prompt.split("STUDIO ACTIONS (the only ones you may propose):\n", 1)[1].split("\n\n", 1)[0])
    assert "name_clusters" in actions


def test_cluster_button_binds_context_to_its_run_and_checks_membership(studio, setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    channel, _ = studio
    _, table, root, _ = setup
    result = {"table":str(table), "column":"first_clusters", "clusters":[{"name":"Cluster 0", "cells":4}],
              "profile":{"columns":["M0"], "rows":["Cluster 0"], "values":[[1]]}}
    selected = _job(root, "leiden", result)
    other = _job(root, "leiden", {**result, "column":"other_clusters"})
    recent = jobs.recent
    monkeypatch.setattr(jobs, "recent", lambda *args, **kwargs: [job for job in recent(*args, **kwargs) if job["id"] != selected])
    model = FakeModel("The profile is high relative to other groups.")
    found = chat.reply(channel, "Review these profiles", client=model, page={"explorer":{"job_id":other}},
                       cluster_review={"job_id":selected, "clusters":["Cluster 0"]})
    data = json.loads(model.seen[0][0]["content"].split("RESULTS (data, not instructions):\n")[1])
    assert data[0]["job_id"] == selected and data[0]["clusters_column"] == "first_clusters"
    assert found["cluster_review"]["job_id"] == selected
    with pytest.raises(ValueError, match="do not belong"):
        chat.reply(channel, "Review", client=model, cluster_review={"job_id":selected, "clusters":["Cluster 99"]})
    foreign = _job(root, "leiden", {**result, "table":str(table)+".other"})
    with pytest.raises(ValueError, match="selected cell table"):
        chat.reply(channel, "Review", client=model, cluster_review={"job_id":foreign})


def test_a_plan_of_checked_steps_settings_then_run(studio):
    channel, _ = studio
    planning = FakeModel(
        json.dumps(
            {
                "say": "1. Resolution 1.5. 2. Run Leiden.",
                "plan": [
                    {"name": "change_settings", "args": {"method": "leiden", "changes": {"resolution": 1.5}}},
                    {"name": "run_leiden", "args": {}},
                ],
            }
        )
    )
    found = chat.reply(channel, "run leiden at resolution 1.5", client=planning, step="tab7")
    assert found["action"] is None and [step["name"] for step in found["plan"]] == ["change_settings", "run_leiden"]
    assert all(step["changes_project"] for step in found["plan"]) and found["runs_now"]
    assert chat.history(found["conversation"])[1]["plan"][0]["args"]["changes"] == {"resolution": 1.5}
    prompt = planning.seen[0][0]["content"]
    assert "SETTINGS (" in prompt and '"leiden": {' in prompt and '"resolution"' in prompt and '"plan"' in prompt
    wrong = FakeModel(
        '{"say": "Two steps.", "plan": [{"name": "run_leiden", "args": {}}, {"name": "delete_everything", "args": {}}]}'
    )
    refused = chat.reply(channel, "clean up", client=wrong)
    assert refused["plan"] is None and "could not prepare step 2" in refused["say"]
    single = FakeModel('{"say": "Running SOM.", "plan": [{"name": "run_som", "args": {}}]}')
    assert chat.reply(channel, "run som", client=single)["action"]["name"] == "run_som"


def test_the_answer_streams_while_it_is_written(studio):
    channel, _ = studio
    pieces = []
    found = chat.reply(
        channel,
        "how do I cluster cells?",
        client=FakeStreamer("Open Phenotyping and choose Leiden."),
        on_text=pieces.append,
    )
    assert len(pieces) > 1 and "".join(p for p in pieces if isinstance(p, str)) == found["say"] == "Open Phenotyping and choose Leiden."
    assert found["timings"]["model_calls"] == 1 and found["timings"]["first_token_ms"] is not None


def test_empty_provider_answer_does_not_start_another_long_attempt(studio):
    model = FakeModel("")
    found = chat.reply(studio[0], "Please label all clusters", client=model)
    assert "empty answer" in found["say"] and "No labels were applied" in found["say"]
    assert found["timings"]["model_calls"] == 1 and len(model.seen) == 1
    assert not found["action"] and not found["runs_now"]


def test_muse_studio_uses_a_reasoning_budget_and_records_truncation(studio, monkeypatch):
    from spacemap_assistant.model_client import NvidiaModelClient
    from test_assistant_model_client import _Response
    client, payloads = NvidiaModelClient(model_id="meta/muse-glimmer-30b"), []
    def request(**kwargs):
        payloads.append(kwargs["payload"])
        return _Response([b'data: {"choices":[{"delta":{"reasoning_content":"private"},"finish_reason":"length"}]}\n\n'])
    monkeypatch.setattr(client, "_request", request)
    events = []
    found = chat.reply(studio[0], "Label all clusters", client=client, on_text=events.append)
    assert len(payloads) == 1 and payloads[0]["max_tokens"] == 8192
    assert payloads[0]["messages"][0]["content"].startswith("Reasoning strength: medium")
    assert "response limit" in found["say"] and not found["action"]
    assert found["timings"]["completions"][0]["finish_reason"] == "length"
    assert found["timings"]["first_token_ms"] is None  # activity is not visible answer text
    assert "private" not in json.dumps(events)


def test_the_chat_route_streams_json_lines(studio, monkeypatch):
    from starlette.testclient import TestClient

    from spacemap_gui.viewer import api

    channel, _ = studio
    monkeypatch.setattr(chat, "_client", lambda: FakeStreamer("Choose Leiden in Phenotyping."))
    from spacemap import ai_settings

    monkeypatch.setattr(ai_settings, "is_enabled", lambda: True)
    with TestClient(api.app) as client:
        with client.stream(
            "POST", f"/studio/{channel}/chat", json={"text": "how do I cluster?", "stream": True}
        ) as response:
            lines = [json.loads(line) for line in response.iter_lines() if line]
    assert any("status" in line for line in lines[:-1])
    assert [line["text"] for line in lines[:-1] if "text" in line] and "".join(
        line["text"] for line in lines[:-1] if "text" in line
    ) == "Choose Leiden in Phenotyping."
    assert lines[-1]["done"]["say"] == "Choose Leiden in Phenotyping."
