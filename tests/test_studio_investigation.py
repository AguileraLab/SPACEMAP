"""Real table/crop tools with bounded, provider-independent model conversations."""

import base64
import copy
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from spacemap import dataio, pending_columns
from spacemap_gui.studio import chat, investigation
from spacemap_gui.viewer import registry
from test_studio_commands import setup  # noqa: F401
from test_studio_chat import studio, _job  # noqa: F401


@pytest.fixture
def study(studio, setup):  # noqa: F811
    channel, store = studio
    _, table, root, spec = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = [101, 102, 103, 104, 201]
    frame["clusters"] = ["Cluster 0", "Cluster 0", "Cluster 1", "Cluster 1", "Cluster 0"]
    frame["area"] = [20, 21, 22, 23, 24]
    frame["M0_corr"] = [0.1, 0.2, 0.3, 0.4, 1.0]
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    job = _job(root, "leiden", {"table":str(table), "column":"clusters", "phenotype_column":"phenotype_leiden",
        "markers":["M0", "M1"], "clusters":[{"name":"Cluster 0", "cells":3}, {"name":"Cluster 1", "cells":2}],
        "profile":{"version":2, "columns":["M0", "M1"], "rows":["Cluster 0", "Cluster 1"], "values":[[1,-1],[-1,1]],
                   "raw":[[2,3],[4,5]], "note":"Arcsinh medians compared across clusters"}})
    page = {"explorer":{"job_id":job, "picked":["Cluster 0"]}}
    return channel, store, table, root, page, spec


def tools(study, **kwargs):
    return investigation.Investigation(study[0], registry.session_project(study[0]), study[4], **kwargs)


def test_sampling_links_global_ids_roi_labels_and_exact_measurements(study):
    helper = tools(study, images=True)
    before = study[2].read_bytes()
    record, images = helper.run("sample_cells", {"count":3, "markers":["M0", "M1"]})
    data = record["data"]
    assert data["matched_cells"] == 3 and data["sample_size"] == 3
    assert data["roi_counts"] == {"P11_roi_2":2, "other":1} and data["sampled_strata"] == 2
    cells = {c["cell_id"]:c for c in data["cells"]}
    assert set(cells) == {101, 102, 201}
    assert cells[101]["local_label"] == 1 and cells[201]["local_label"] == 1
    assert cells[102]["measurements"]["M0"] == 2 and cells[102]["measurements"]["M0_corr"] == .2
    assert cells[101]["features_and_labels"]["area"] == 20
    assert cells[101]["cluster"] == "Cluster 0" and cells[101]["cluster_column"] == "clusters"
    assert "No unique ROI image" in cells[201]["image_note"]
    assert cells[101]["crop"]["image"] == "P11_roi_2.ome.tiff" and "centroid cross" in cells[101]["crop"]["locator"]
    assert images and record["images_available_to_model"]
    with Image.open(io.BytesIO(base64.b64decode(images[0].split(",")[1]))) as sheet:
        assert sheet.size == (444, 380)
    assert study[2].read_bytes() == before
    assert not pending_columns.pending_names(study[2])


def test_group_summaries_cover_all_matches_not_just_samples(study):
    record, images = tools(study).run("summarize_cells", {"columns":["M0", "batch_id"], "filters":[{"col":"clusters", "values":["Cluster 0"]}]})
    summary = record["data"]["summaries"]
    assert summary[0]["count"] == 3 and summary[0]["median"] == 2
    assert summary[0]["mean"] == pytest.approx(103/3) and summary[1]["distinct"] == 2 and not images


def test_two_cluster_samples_share_one_measurement_read(study, monkeypatch):
    helper = tools(study)
    reads, take = [], helper.table.take
    def track(rows, columns):
        reads.append(list(rows))
        return take(rows, columns)
    monkeypatch.setattr(helper.table, "take", track)
    records, images = helper.sample_pair([{"clusters":[n], "count":6, "images":False} for n in (0,1)])
    assert len(reads) == 1 and set(reads[0]) == set(range(5))
    assert [r["data"]["matched_cells"] for r in records] == [3,2]
    assert [{c["cluster"] for c in r["data"]["cells"]} for r in records] == [{"Cluster 0"},{"Cluster 1"}]
    assert not images and helper.sampled == 5


def test_transforms_share_the_viewer_and_corrected_values_stay_separate(study):
    record, _ = tools(study, view={"cells":{"marker_transforms":["Arcsinh", "Z2"]}}).run("sample_cells", {"count":3, "images":False})
    cells = {c["cell_id"]:c for c in record["data"]["cells"]}
    assert cells[101]["transforms"] == ["Arcsinh", "Z2"]
    assert cells[101]["transformed_raw"]["M0"] == pytest.approx(-1)
    assert cells[101]["transformed_raw"]["M1"] == pytest.approx(1)
    assert cells[101]["measurements"]["M0_corr"] == .1


def test_reinspection_is_restricted_and_table_changes_invalidate_review(study):
    helper = tools(study)
    with pytest.raises(ValueError, match="rows returned"):
        helper.run("inspect_cells", {"rows":[0]})
    first, _ = helper.run("sample_cells", {"count":2, "images":False})
    rows = [c["row"] for c in first["data"]["cells"]]
    second, _ = helper.run("inspect_cells", {"rows":rows, "images":False})
    assert [c["row"] for c in second["data"]["cells"]] == rows
    study[2].write_text(study[2].read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        helper.run("describe_data", {})


@pytest.mark.parametrize("name,args", [
    ("exec", {"cmd":"anything"}), ("describe_data", {"path":"/tmp/another-table.csv"}),
    ("sample_cells", {"count":500}), ("sample_cells", {"seed":-1}),
    ("sample_cells", {"clusters":["Cluster 99"]}),
    ("summarize_cells", {"columns":["missing"]}),
    ("sample_cells", {"filters":[{"col":"missing", "values":["x"]}]}),
])
def test_invalid_requests_cannot_read_other_data(study, name, args):
    with pytest.raises(ValueError):
        tools(study).run(name, args)


class Model:
    def __init__(self, answers):
        self.answers, self.seen = iter(answers), []
    def complete(self, messages, **kwargs):
        self.seen.append(copy.deepcopy(messages))
        return next(self.answers)


@pytest.mark.parametrize("vision", [True, False])
def test_chat_collects_real_evidence_before_answer_for_either_model(study, monkeypatch, vision):
    monkeypatch.setattr(chat, "sees_images", lambda: vision)
    model = Model([
        json.dumps({"tools":[{"name":"sample_cells", "args":{"count":3, "markers":["M0", "M1"]}}]}),
        json.dumps({"tools":[{"name":"summarize_cells", "args":{"columns":["M0"], "filters":[{"col":"clusters", "values":["Cluster 0"]}]}}]}),
        "Three sampled cells from two ROIs differ in expression. One ROI has no image, so this is tentative.",
    ])
    progress = []
    answer = chat.reply(study[0], "Compare the expression and visual evidence of this cluster", page=study[4], client=model, on_text=progress.append)
    assert len(answer["investigation"]) == 2 and answer["evidence_images"]
    assert answer["action"] is None and not answer["runs_now"]
    assert any(isinstance(event, dict) and "status" in event for event in progress)
    second = model.seen[1][-1]["content"]
    if vision:
        assert isinstance(second, list) and second[1]["type"] == "image_url"
    else:
        assert isinstance(second, str) and "cannot inspect images" in answer["say"]
    assert "101" in json.dumps(second) and "M0_corr" in json.dumps(second)
    history = chat.history(answer["conversation"])
    assert history[-1]["investigation"][0]["data"]["sample_size"] == 3
    assert "data:image" not in json.dumps(history)  # no pictures in the conversation database


def test_tool_errors_are_returned_as_evidence_and_never_success(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: False)
    model = Model(['{"tools":[{"name":"read_file","args":{"path":"/etc/passwd"}}]}', "That file tool is unavailable."])
    answer = chat.reply(study[0], "Inspect the data", page=study[4], client=model)
    assert answer["investigation"][0]["error"] and not answer["evidence_images"]
    assert "error" in model.seen[1][-1]["content"]


def test_investigation_does_not_authorize_label_edits(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: False)
    model = Model(['{"tools":[{"name":"sample_cells","args":{"count":1,"images":false}}]}',
                   json.dumps({"say":"A possible T cell.", "action":{"name":"name_clusters", "args":{"job_id":study[4]["explorer"]["job_id"], "names":{"Cluster 0":"T"}}}})])
    answer = chat.reply(study[0], "What might this cluster be? Inspect examples.", page=study[4], client=model)
    assert answer["action"] is None and not answer["runs_now"]


def test_explicit_sampling_fetches_evidence_before_the_first_model_call(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: True)
    model = Model(["Two cells inspected; their measurements differ."])
    found = chat.reply(study[0], "Pick 2 cells from Cluster 0 and review their marker tiles", page=study[4], client=model)
    assert len(model.seen) == 1 and found["timings"]["model_calls"] == 1
    assert found["investigation"][0]["data"]["sample_size"] == 2
    content = model.seen[0][-1]["content"]
    assert content[0]["text"].startswith("TOOLS RESULTS") and content[1]["type"] == "image_url"


def test_missing_profile_reply_is_repaired_from_the_selected_run(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: False)
    job = study[4]["explorer"]["job_id"]
    model = Model(["Please provide the cluster marker profiles.", json.dumps({"say":"Names for review.",
        "action":{"name":"name_clusters", "args":{"job_id":job, "names":{"Cluster 0":"T", "Cluster 1":"B"}}}})])
    found = chat.reply(study[0], "Suggest names for all clusters", mode="suggest", page=study[4], client=model,
                       cluster_review={"job_id":job, "clusters":[]})
    assert len(model.seen) == 2 and found["action"]["args"]["names"] == {"Cluster 0":"T", "Cluster 1":"B"}
    assert not found["runs_now"]
    evidence = found["investigation"][0]
    assert evidence["tool"] == "cluster_profiles" and evidence["data"]["profile"]["medians_before_cluster_zscore"] == [[2,3],[4,5]]
    assert "not necessarily raw camera intensities" in model.seen[1][-1]["content"]
    assert found["cluster_review"] == {"job_id":job, "clusters":2, "markers":2, "images":0}


def test_image_budget_and_tool_budget_bound_the_conversation(study):
    helper = tools(study, images=True, image_budget=1)
    first, images = helper.run("sample_cells", {"count":2})
    again, later = helper.run("inspect_cells", {"rows":[c["row"] for c in first["data"]["cells"]]})
    assert len(images) == 1 and later == [] and not again["images_available_to_model"]
    one = {"name":"describe_data", "args":{}}
    sequence = iter([json.dumps({"tools":[one,one]})] * 3 + ["Done reading metadata."])
    _, records, _ = investigation.answer(json.dumps({"tools":[one]}), channel=study[0],
        project=registry.session_project(study[0]), page=study[4], view={}, sees_images=False,
        messages=[], complete=lambda history: next(sequence))
    assert len(records) == 6
    limited = tools(study, cell_budget=0)
    with pytest.raises(ValueError, match="inspection limit"):
        limited.run("sample_cells", {"count":1})
    raw, records, _ = investigation.answer(json.dumps({"tools":[one]}), channel=study[0],
        project=registry.session_project(study[0]), page=study[4], view={}, sees_images=False,
        messages=[], complete=lambda history: pytest.fail("No budget remains"), tool_budget=0)
    assert not records and "limit was reached" in raw


def test_empty_population_is_reported_without_fabricated_samples(study):
    record, images = tools(study).run("sample_cells", {"filters":[{"col":"batch_id", "values":["absent"]}]})
    assert record["data"]["matched_cells"] == 0 and record["data"]["cells"] == [] and not images


PASTED_CALLS = '<|tool_call>call:spacemap_ai:sample_cells{job_id: "81fe69ebf58a4f15a9a9c255cc928d0",clusters:[4],count:6}<tool_call|><|tool_call>call:spacemap_ai:sample_cells{job_id: "81fe69ebf58a4f15a9a9c255cc928d0",clusters:[25],count:6}<tool_call|>'


def test_exact_pasted_tool_calls_are_data_not_an_answer():
    calls = investigation.request(PASTED_CALLS)
    assert len(calls) == 2
    assert [call["args"]["clusters"] for call in calls] == [[4], [25]]
    assert all(call["name"] == "sample_cells" and call["args"]["count"] == 6 for call in calls)


@pytest.mark.parametrize("raw", [
    '<|tool_call>call:spacemap_ai:sample_cells{count:__import__("os").getcwd()}<tool_call|>',
    '<|tool_call>call:spacemap_ai:sample_cells{count:1,count:6}<tool_call|>',
    '<|tool_call>call:spacemap_ai:sample_cells{count:',
    '{"tools": [unfinished',
])
def test_invalid_protocol_is_never_treated_as_prose(raw):
    assert investigation.request(raw) == []


def test_native_parser_preserves_quoted_measurement_names():
    raw = '<|tool_call>call:spacemap_ai:summarize_cells{columns:["x,{marker: 2}","a\\\"b"],filters:[]}<tool_call|>'
    assert investigation.request(raw)[0]["args"] == {"columns":["x,{marker: 2}", 'a"b'], "filters":[]}


def test_numeric_cluster_ids_and_shortened_id_bind_to_verified_run(study):
    job = study[4]["explorer"]["job_id"]
    for name in ("sample_cells", "cluster_profiles"):
        record, _ = tools(study).run(name, {"job_id":job[:-1], "clusters":[0]})
        assert record["args"]["job_id"] == job and record["args"]["clusters"] == ["Cluster 0"]
    with pytest.raises(ValueError, match="different run"):
        tools(study).run("sample_cells", {"job_id":"f"*31, "clusters":[0]})
    with pytest.raises(ValueError, match="not found"):
        tools(study).run("sample_cells", {"job_id":"f"*32, "clusters":[0]})
    with pytest.raises(ValueError, match="Unknown cluster"):
        tools(study).run("sample_cells", {"clusters":[99]})
    with pytest.raises(ValueError, match="cluster IDs"):
        tools(study).run("sample_cells", {"clusters":[True]})


def test_native_requests_run_tools_then_return_prose_without_stream_leaks(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: True)
    job = study[4]["explorer"]["job_id"]
    raw = PASTED_CALLS.replace('81fe69ebf58a4f15a9a9c255cc928d0', job[:-1]).replace('[4]', '[0]').replace('[25]', '[1]')
    class NativeModel(Model):
        def stream_complete(self, messages, **kwargs):
            answer = self.complete(messages, **kwargs)
            for char in answer:  # Even one-character network fragments cannot leak a prefix.
                yield char
    model = NativeModel([raw, "Reviewed both groups. The sample is too small to establish their identity."])
    events = []
    found = chat.reply(study[0], "Compare the visual evidence for these two groups", page=study[4], client=model, on_text=events.append)
    assert len(found["investigation"]) == 2 and len(model.seen) == 2
    assert [r["data"]["sample_size"] for r in found["investigation"]] == [3, 2]
    last = model.seen[-1][-1]["content"]
    assert len(last) == 3 and all(part["type"] == "image_url" for part in last[1:])
    streamed = "".join(part for part in events if isinstance(part, str))
    assert streamed == found["say"] and "tool_call" not in streamed
    assert not found["action"] and not found["runs_now"]


def test_unusable_tool_output_returns_readable_failure_without_labels(study):
    model = Model(['<|tool_call>broken', '<|tool_call>still broken'])
    found = chat.reply(study[0], "Compare this group", page=study[4], client=model)
    assert "could not collect" in found["say"] and "tool_call" not in found["say"]
    assert len(found["investigation"]) == 2 and all(r.get("error") for r in found["investigation"])
    assert not found["action"] and not found["runs_now"] and len(model.seen) == 2


def test_typo_and_conversational_cluster_scope_fetch_both_groups_first(study, monkeypatch):
    monkeypatch.setattr(chat, "sees_images", lambda: True)
    channel, store, _, root, page, _ = study
    page = {"explorer":{**page["explorer"], "picked":[]}}
    conversation = store.create_conversation(title="Review", project_root=str(root))
    store.add_message(conversation, "user", "Are you sure cluster 0 and 1 should be Others?", metadata={"review_job_id":page["explorer"]["job_id"]})
    store.add_message(conversation, "assistant", "Cluster 99 is mixed.")  # Never take scope from model guesses.
    model = Model(["Both groups were sampled. These are tentative observations."])
    found = chat.reply(channel, "Ok, you still need to test marker tiles for smple of cells from these clusters and decide on what to call them, can you do that?",
                       conversation=conversation, page=page, client=model)
    assert len(model.seen) == 1 and len(found["investigation"]) == 2
    assert [r["args"]["clusters"] for r in found["investigation"]] == [["Cluster 0"], ["Cluster 1"]]
    assert not found["action"] and not found["runs_now"]


def test_sampling_scope_does_not_cross_runs_or_ignore_explicit_clusters():
    page = {"explorer":{"job_id":"a"*32, "picked":[]}}
    history = [{"role":"user", "content":"Cluster 4 and 25", "metadata":{"review_job_id":"b"*32}}]
    assert investigation.sample_request("Test marker tiles for cells from these clusters", page, history) is None
    explicit = investigation.sample_request("Inspect 3 cells from clusters 4 and 25", page, history)
    assert [t["args"]["clusters"] for t in explicit["tools"]] == [["Cluster 4"], ["Cluster 25"]]
    assert all(t["args"]["count"] == 3 for t in explicit["tools"])
    for text in ("How do I sample cells?", "Do not sample cells", "Inspect cells from clusters 1, 2 and 3"):
        assert investigation.sample_request(text, page) is None


def test_last_contact_sheet_keeps_all_cells_when_view_uses_a_slot(study):
    helper = tools(study, images=True, image_budget=3)
    tile = Image.new("RGB", (144,144), "black")
    def batch():
        return [({"row":i,"cell_id":i,"roi":"roi","local_label":i}, [tile], ["M0"]) for i in range(6)]
    first, second = batch(), batch()
    helper._sheets(first)
    helper._sheets(second)
    assert len(helper.images) == 3
    assert all(row[0]["images_available_to_model"] for row in [*first, *second])
    with Image.open(io.BytesIO(base64.b64decode(helper.images[-1].split(",")[1]))) as sheet:
        assert sheet.height == 1140
