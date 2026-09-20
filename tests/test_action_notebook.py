import json
from pathlib import Path

from spacemap_assistant import action_notebook


def test_action_notebook_keeps_plan_results_table_preview_and_log(tmp_path, monkeypatch) -> None:
    preview = tmp_path / "qc.png"
    # A valid 1 x 1 transparent PNG.
    preview.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
        )
    )
    table = tmp_path / "comparison.csv"
    table.write_text("trial,score\na,0.9\n", encoding="utf-8")
    details = {
        "directory": str(tmp_path),
        "plan": {
            "action_id": "00000000-0000-0000-0000-000000000001",
            "action_type": "optimize_artifact_detection",
            "created_at": "2026-09-06T12:00:00+00:00",
            "user_request": "Test bright artifacts.",
            "run_output_dir": str(tmp_path),
        },
        "status": {"state": "completed", "stage": "Complete"},
        "report": {
            "result": {
                "comparison": [{"trial": "a", "score": 0.9}],
                "outputs": {"comparison_csv": str(table)},
                "review_previews": [{"preview_path": str(preview)}],
            }
        },
    }
    monkeypatch.setattr(action_notebook, "action_details", lambda _action_id: details)
    monkeypatch.setattr(action_notebook, "progress_events", lambda _action_id, limit=2000: [])
    monkeypatch.setattr(action_notebook, "action_log", lambda _action_id, max_characters=200000: "finished")
    monkeypatch.setattr(action_notebook, "is_assistant_read_allowed", lambda _path: True)

    document = action_notebook.build_action_notebook(details["plan"]["action_id"])

    assert document["nbformat"] == 4
    assert document["metadata"]["spacemap_ai"]["action_id"] == details["plan"]["action_id"]
    serialized = json.dumps(document)
    assert "Test bright artifacts" in serialized
    assert "comparison.csv" in serialized
    assert "image/png" in serialized
    assert "finished" in serialized
    assert action_notebook.notebook_action_id(serialized) == details["plan"]["action_id"]


def test_notebook_title_uses_real_action_type() -> None:
    assert action_notebook.action_title({"action_type": "register_codex_he_pair"}) == "CODEX + H&E registration"
    assert action_notebook.action_title({"action_type": "extract_rois"}) == "ROI extraction"
    assert action_notebook.action_title({"action_type": "optimize_blur_detection"}) == "Blur parameter test"
