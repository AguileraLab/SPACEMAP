from types import SimpleNamespace

from spacemap_assistant import blur_worker
from spacemap_workflows.annotation import default_blur_trials, default_bright_trials


def test_default_blur_trials_test_population_separation_without_exceeding_limit() -> None:
    trials = default_blur_trials({})
    assert len(trials) == 10
    relaxed = [
        item for item in trials
        if item.get("overrides", {}).get("blur_min_separation") is not None
    ]
    assert {item["overrides"]["blur_min_separation"] for item in relaxed} == {2.0, 2.5}
    assert any(item["overrides"].get("blur_evidence_rule") == "pooled_only" for item in relaxed)


def test_default_bright_trials_cover_brightness_and_texture_without_exceeding_limit() -> None:
    trials = default_bright_trials({})
    assert len(trials) == 10
    assert trials[0]["name"] == "current_project_settings"
    combinations = {
        (
            item["overrides"].get("bright_flat_mean_percentile"),
            item["overrides"].get("bright_flat_texture_percentile"),
        )
        for item in trials[1:]
    }
    assert combinations == {
        (brightness, texture)
        for brightness in (99.0, 99.5, 99.8)
        for texture in (1.0, 3.0, 5.0)
    }


def test_automatic_local_visual_review_retries_one_empty_answer(tmp_path, monkeypatch) -> None:
    preview = tmp_path / "candidate.png"
    preview.write_bytes(b"small generated png")
    calls: list[list[dict]] = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def complete(self, messages, **_kwargs):
            calls.append(messages)
            return "" if len(calls) == 1 else "Candidate 1: likely false positive; structured tissue."

    monkeypatch.setattr(blur_worker, "LocalModelClient", FakeClient)
    monkeypatch.setattr(
        blur_worker,
        "load_model_settings",
        lambda: SimpleNamespace(timeout_seconds=30, max_tokens=512),
    )
    plan = SimpleNamespace(
        automatic_visual_review=True,
        online_visual_review_approved=False,
        provider="local",
        model_id="google/gemma-4-12B-it",
        artifact_kind="bright",
    )
    result = {
        "selected_trial": "trial_1",
        "comparison": [
            {
                "trial": "trial_1",
                "polygon_count": 1,
                "bright_percent": 0.1,
                "brightness_percentile": 99.5,
                "texture_percentile": 3.0,
            }
        ],
    }
    updates = []
    review = blur_worker._automatic_visual_review(
        plan,
        result,
        [{"preview_path": str(preview), "qc_modality": "he", "qc_kind": "candidate_tile"}],
        lambda stage, message: updates.append((stage, message)),
    )
    assert review["state"] == "completed"
    assert "likely false positive" in review["answer"]
    assert len(calls) == 2
    assert any("retrying" in message for _stage, message in updates)


def test_online_visual_review_waits_when_the_run_was_not_approved(tmp_path) -> None:
    preview = tmp_path / "candidate.png"
    preview.write_bytes(b"small generated png")
    plan = SimpleNamespace(
        automatic_visual_review=True,
        online_visual_review_approved=False,
        provider="nvidia",
        model_id="meta/muse-glimmer-30b",
        artifact_kind="bright",
    )
    review = blur_worker._automatic_visual_review(
        plan,
        {"selected_trial": "", "comparison": []},
        [{"preview_path": str(preview)}],
        lambda _stage, _message: None,
    )
    assert review["state"] == "waiting_for_online_approval"
