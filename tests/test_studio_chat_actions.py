"""Executable naming requests, proposal recovery, receipts and reversible edits."""

import json
from pathlib import Path

import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import chat, chat_actions, cluster_names, jobs, leiden
from test_studio_chat import FakeModel, _job, studio  # noqa: F401
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def naming(studio, setup, monkeypatch):  # noqa: F811
    channel, store = studio
    _, table, root, _ = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = range(len(frame))
    frame["clusters"] = ["Cluster 0"] * 2 + ["Cluster 1"] * 3
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    job_id = _job(root, "leiden", {"table":str(table), "column":"clusters", "phenotype_column":"phenotype_leiden", "markers":["M0","M1"],
        "clusters":[{"name":"Cluster 0", "cells":2}, {"name":"Cluster 1", "cells":3}]})
    calls = []
    def start(root, kind, settings, **kwargs):
        root = Path(root)
        calls.append(settings)
        named_id = _job(root, kind, {}, settings)
        result = leiden.name_clusters(settings, jobs.directory(root, named_id), lambda *args: None)
        path = jobs.directory(root, named_id) / "status.json"
        status = json.loads(path.read_text())
        status["result"] = result
        path.write_text(json.dumps(status))
        return status
    monkeypatch.setattr(jobs, "start", start)
    return channel, store, root, table, job_id, calls


def apply(naming, answer, args=None):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    command = answer["action"]
    response = TestClient(app).post(f"/studio/{naming[0]}/command", json={
        **command, **({"args":args} if args is not None else {}), "confirmed":True,
        "assistant":{"conversation":answer["conversation"], "message":answer["message_id"]}})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "done", response.text
    return response.json()


def suggestion(naming, **kwargs):
    action = {"name":"name_clusters", "args":{"job_id":naming[4], "names":{"Cluster 0":"CD4", "Cluster 1":"Others"}}}
    return chat.reply(naming[0], "Suggest names for these clusters", mode="suggest", client=FakeModel(json.dumps({"action":action})),
                      page={"explorer":{"job_id":naming[4]}}, **kwargs)


def test_malformed_plan_closer_is_recovered_without_inventing_values():
    raw = 'I will label them. ```json\n{"say":"Names","plan":[{"name":"name_clusters","args":{"job_id":"a","names":{"Cluster 0":"CD4","Cluster 1":"Others"}}}}\n```'
    _, action, plan = chat.read_reply(raw)
    assert action["args"]["names"] == {"Cluster 0":"CD4", "Cluster 1":"Others"} and plan is None
    for broken in ['{"action":{"name":"name_clusters","args":{"names":{"Cluster 1":"Ot',
                   '{"action":{"name":"name_clusters","args":']:
        assert chat.read_reply(broken)[1:] == (None, None)


def test_missing_outer_closer_keeps_complete_names_without_a_second_model_call():
    payload = {"say":"Review these names", "action":{"name":"name_clusters", "args":{
        "job_id":"saved-job", "names":{"Cluster 0":"CD4", "Cluster 1":"Others"}}}}
    _, action, plan = chat.read_reply(json.dumps(payload)[:-1])
    assert action["args"]["names"] == payload["action"]["args"]["names"] and plan is None
    # Never manufacture missing values, commas, or quotes.
    for suffix in [',', ':', '"Others', '12']:
        assert chat.read_reply('{"action":{"name":"name_clusters","args":{"job_id":"a","names":{"Cluster 0":' + suffix)[1:] == (None, None)


@pytest.mark.parametrize("form", ["keyed", "flat", "bare", "nested"])
def test_provider_naming_forms_fill_drafts_in_one_call(naming, form):
    args = {"job_id":naming[4][:-1], "names":{"Cluster 0":"CD4", "Cluster 1":"Others"}}
    value = {"keyed":{"name_clusters":args}, "flat":{"action":"name_clusters", "args":args},
             "bare":{"name":"name_clusters", "args":args},
             "nested":{"action":{"name":"name_clusters", "args":args}}}[form]
    model = FakeModel('Suggested names for review.\n```json\n' + json.dumps(value) + '\n```')
    found = chat.reply(naming[0], "Suggest names for all clusters", mode="suggest", client=model,
                       cluster_review={"job_id":naming[4]})
    assert found["action"]["args"] == {**args, "job_id":naming[4]}
    assert len(model.seen) == 1 and found["timings"]["model_calls"] == 1
    assert not found["runs_now"] and not naming[5]


def test_flat_naming_request_saves_and_repaints_through_command_bus(naming):
    raw = {"action":"name_clusters", "args":{"job_id":naming[4][:-1],
           "names":{"Cluster 0":"Epithelia", "Cluster 1":"Others"}}}
    model = FakeModel(json.dumps(raw))
    found = chat.reply(naming[0], "Label these clusters", client=model, page={"explorer":{"job_id":naming[4]}})
    assert found["runs_now"] and len(model.seen) == 1
    assert apply(naming, found)["status"] == "done"
    assert dataio.load(naming[3], columns=["phenotype_leiden"])["phenotype_leiden"].tolist() == ["Epithelia"]*2 + ["Others"]*3


def test_naming_without_retyped_id_uses_bound_run_only(naming):
    raw = json.dumps({"name_clusters":{"names":{"Cluster 0":"CD4"}}})
    assert chat.read_reply(raw)[1] is None  # no context, no guessed run
    found = chat.reply(naming[0], "Suggest names", mode="suggest", client=FakeModel(raw), cluster_review={"job_id":naming[4]})
    assert found["action"]["args"]["job_id"] == naming[4]


@pytest.mark.parametrize("bad_id", ["1"*32, "unrelated", "1"*31])
def test_binding_never_redirects_another_run(naming, bad_id):
    _, action, _ = chat.read_reply(json.dumps({"name_clusters":{"job_id":bad_id,"names":{"Cluster 0":"CD4"}}}), cluster_job_id=naming[4])
    assert action is None


def test_name_the_cluster_then_recovers_previous_reply_without_inference(naming, monkeypatch):
    conversation = naming[1].create_conversation(title="Reported naming failure", project_root=str(naming[2]))
    raw = 'No change was made: the assistant did not return a valid Studio action.\n```json\n' + json.dumps({
        "name_clusters":{"job_id":naming[4][:-1], "names":{"Cluster 0":"CD4","Cluster 1":"Others"}}}) + '\n```'
    naming[1].add_message(conversation, "assistant", raw, metadata={"project":str(naming[2]),"cluster_review":{"job_id":naming[4]}})
    monkeypatch.setattr(chat, "_client", lambda: pytest.fail("Applying existing names must not call the model"))
    found = chat.reply(naming[0], "Name the cluster then", conversation=conversation, page={"explorer":{"job_id":naming[4]}})
    assert found["runs_now"] and found["timings"]["model_calls"] == 0
    assert apply(naming, found)["status"] == "done"
    assert naming[5][-1]["names"] == {"Cluster 0":"CD4","Cluster 1":"Others"}


def test_normalization_does_not_accept_unknown_commands_or_invalid_arguments():
    assert chat.read_reply('{"delete_everything":{}}')[1] is None
    assert chat.read_reply('{"action":"delete_everything","args":{}}')[1] is None
    assert chat.read_reply('{"name_clusters":{"job_id":"a","names":"CD4"}}')[1] is None


def test_suggestions_are_drafts_but_conversational_approval_runs_once(naming):
    first = suggestion(naming)
    assert first["action"] and not first["runs_now"] and not naming[5]
    approved = chat.reply(naming[0], "ok, do that", conversation=first["conversation"])
    assert approved["source"] == "instant" and approved["runs_now"] and approved["action"] == first["action"]
    result = apply(naming, approved)
    assert result["status"] == "done" and len(naming[5]) == 1
    values = dataio.load(naming[3], columns=["phenotype_leiden"])["phenotype_leiden"].tolist()
    assert values == ["CD4", "CD4", "Others", "Others", "Others"]
    assert apply(naming, approved)["replayed"] and len(naming[5]) == 1
    # The older proposal card also resolves to the same command receipt.
    assert apply(naming, first)["replayed"] and len(naming[5]) == 1
    again = chat.reply(naming[0], "ok, do that", conversation=approved["conversation"])
    assert again["action"] is None and "Already saved in phenotype_leiden" in again["say"]
    assert any(item["execution"] and item["execution"]["status"] == "done" for item in chat.history(first["conversation"]))


def test_direct_label_changes_and_clearing_preserve_other_clusters(naming):
    apply(naming, suggestion(naming))  # a user can also approve via the card
    page = {"explorer":{"job_id":naming[4]}}
    edit = chat.reply(naming[0], "Rename Cluster 0 to Treg", page=page)
    assert edit["runs_now"] and edit["source"] == "instant"
    assert apply(naming, edit)["status"] == "done"
    assert naming[5][-1]["names"] == {"Cluster 0":"Treg", "Cluster 1":"Others"}
    clear = chat.reply(naming[0], "Remove the label for Cluster 0", page=page)
    assert clear["runs_now"] and apply(naming, clear)["status"] == "done"
    assert naming[5][-1]["names"] == {"Cluster 1":"Others"}
    saved = cluster_names.saved_names(naming[2], str(naming[3]), "clusters")
    values = dataio.load(naming[3], columns=[saved["column"]])[saved["column"]].tolist()
    assert values == ["Cluster 0", "Cluster 0", "Others", "Others", "Others"]
    clear_all = chat.reply(naming[0], "Clear all cluster names", page=page)
    assert apply(naming, clear_all)["status"] == "done" and naming[5][-1]["names"] == {}
    assert "phenotype_leiden" in dataio.columns(naming[3])
    assert "phenotype_leiden_2" not in dataio.columns(naming[3])  # edits keep the chosen column


def test_explicit_naming_request_runs_and_unrelated_job_does_not(naming):
    action = suggestion(naming)["action"]
    answer = chat.reply(naming[0], "Can you write the cluster names into the table?", client=FakeModel(json.dumps({"action":action})))
    assert answer["runs_now"] and answer["say"] == "Applying the cluster names…"
    wrong = chat.reply(naming[0], "Label these clusters", client=FakeModel('{"action":{"name":"run_batch_correction","args":{}}}'))
    assert not wrong["runs_now"]


def test_old_malformed_proposal_can_be_applied_without_another_model_call(naming):
    conversation = naming[1].create_conversation(title="Old naming request", project_root=str(naming[2]))
    action = {"name":"name_clusters", "args":{"job_id":naming[4], "names":{"Cluster 0":"CD4", "Cluster 1":"Others"}}}
    malformed = json.dumps({"plan":[action]})[:-2] + "}}"
    naming[1].add_message(conversation, "assistant", malformed)
    naming[1].add_message(conversation, "assistant", "I have submitted the plan.")
    approved = chat.reply(naming[0], "Apply those names", conversation=conversation)
    assert approved["runs_now"] and approved["source"] == "instant"
    assert apply(naming, approved)["status"] == "done"


def test_a_single_cluster_suggestion_cannot_change_other_clusters(naming):
    action = suggestion(naming)["action"]
    answer = chat.reply(naming[0], "Suggest a name for Cluster 0", mode="suggest", client=FakeModel(json.dumps({"action":action})),
                        page={"explorer":{"job_id":naming[4], "picked":["Cluster 0"]}})
    assert answer["action"]["args"]["names"] == {"Cluster 0":"CD4"}


def test_false_success_is_not_reported_as_an_applied_action(naming):
    model = FakeModel("I have submitted the plan and saved the names.")
    answer = chat.reply(naming[0], "Label these clusters", client=model)
    assert len(model.seen) == 2 and not answer["runs_now"] and not answer["action"]
    assert answer["say"].startswith("No change was made") and "submitted the plan" not in answer["say"]


def test_an_unknown_cluster_cannot_fill_fields_or_run(naming):
    model = FakeModel(json.dumps({"action":{"name":"name_clusters", "args":{"job_id":naming[4], "names":{"Cluster 999":"Treg"}}}}))
    answer = chat.reply(naming[0], "Label these clusters", client=model)
    assert answer["action"] is None and "There is no Cluster 999" in answer["say"]


def test_failed_job_is_not_reported_as_saved(naming):
    answer = suggestion(naming)
    result = apply(naming, answer)
    path = jobs.directory(naming[2], result["data"]["job"]["id"]) / "status.json"
    status = json.loads(path.read_text()); status.update(status="failed", message="Disk full")
    path.write_text(json.dumps(status))
    saved = chat_actions.execution_status(chat_actions.receipt(naming[1], answer["conversation"], answer["message_id"]), naming[2])
    assert saved["status"] == "failed" and saved["message"] == "Disk full"


def test_receipt_keeps_the_names_edited_in_the_review_card(naming):
    answer = suggestion(naming)
    edited = {**answer["action"]["args"], "names":{"Cluster 0":"CD8", "Cluster 1":"Others"}}
    apply(naming, answer, edited)
    saved = chat_actions.receipt(naming[1], answer["conversation"], answer["message_id"])
    assert saved["args"]["names"]["Cluster 0"] == "CD8"


@pytest.mark.parametrize("text", ["How do I label these clusters?", "Suggest names for these clusters", "Do not label these clusters", "What if I rename Cluster 0?"])
def test_questions_and_suggestions_do_not_authorize_writes(text):
    assert not chat_actions.authorizes(text, {"name":"name_clusters", "args":{}})
