import tempfile
from pathlib import Path

from spacemap_assistant.paths import ASSISTANT_DATA_DIR
from spacemap_assistant.blur_action import parse_artifact_optimization_request
from spacemap_assistant.roi_action import parse_roi_extraction_request
from spacemap_assistant.storage import AssistantStore
from spacemap_gui.components.assistant_panel import (
    _advisory_answer_needs_repair,
    _cluster_text_only_messages,
    _artifact_draft_with_thread_context,
    _completed_blur_answer,
    _is_annotation_qc_followup,
    _is_plain_approval,
    _last_user_request,
    _quick_model_label,
    _quick_model_options,
    _recent_annotation_qc_bundle,
    _recent_preview_states,
    _recent_preview_source,
    _request_with_recent_path_context,
)


def test_internal_model_thoughts_are_not_accepted_as_chat_answers() -> None:
    assert _advisory_answer_needs_repair("None")
    assert _advisory_answer_needs_repair(
        "thought call:spacemap_gui.tabs.tab9.build_generic_paper_figure{}"
    )
    assert not _advisory_answer_needs_repair(
        "Cluster 4 is most consistent with an epithelial-rich tissue region."
    )


def test_cluster_fallback_replaces_only_the_latest_user_payload() -> None:
    original = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": [{"type": "image_url"}, {"type": "text"}]},
    ]
    compact = _cluster_text_only_messages(original, "Label clusters", "\nTABLES")
    assert compact[1]["content"] == "older"
    assert "Label clusters\nTABLES" in compact[-1]["content"]
    assert "visual endpoint was unavailable" in compact[-1]["content"]
    assert isinstance(original[-1]["content"], list)


def test_path_only_followup_completes_recent_roi_request() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        conversation = store.create_conversation()
        store.add_message(
            conversation,
            "user",
            "Register /allowed/data/codex/old.qptiff with /allowed/data/he/old.qptiff",
        )
        store.add_message(
            conversation,
            "user",
            "Extract ROIs using /allowed/bounds.geojson and save in /allowed/outputs/rois",
        )
        image = "/allowed/outputs/registered/sample.ome.tiff"
        store.add_message(conversation, "user", image)
        combined = _request_with_recent_path_context(image, store, conversation)
        draft = parse_roi_extraction_request(combined)
        assert draft is not None
        assert draft["image_path"] == image
        assert draft["boundaries_path"] == "/allowed/bounds.geojson"
        assert draft["output_dir"] == "/allowed/outputs/rois"


def test_csv_only_followup_keeps_the_cluster_labeling_question() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        conversation = store.create_conversation()
        original = "Can you label these clusters based on their marker expression?"
        csv_path = "/allowed/figures/step3_cluster_assignments.csv"
        store.add_message(conversation, "user", original)
        store.add_message(conversation, "assistant", "Please provide the table.")
        store.add_message(conversation, "user", csv_path)
        combined = _request_with_recent_path_context(csv_path, store, conversation)
        assert original in combined
        assert csv_path in combined


def test_plain_chat_approval_is_narrow() -> None:
    assert _is_plain_approval("approve")
    assert _is_plain_approval("Yes, approve.")
    assert not _is_plain_approval("Approve every future job automatically")


def test_quick_model_switch_lists_local_and_every_hosted_model() -> None:
    options = _quick_model_options()
    assert options[0] == "local"
    assert all(_quick_model_label(value) for value in options)
    assert any(value.startswith("nvidia:") for value in options)


def test_retry_uses_only_the_latest_user_request_in_the_open_chat() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        current_chat = store.create_conversation()
        other_chat = store.create_conversation()
        store.add_message(current_chat, "user", "First request")
        store.add_message(current_chat, "assistant", "First answer")
        store.add_message(other_chat, "user", "Wrong chat request")
        store.add_message(current_chat, "user", "Newest request")
        store.add_message(current_chat, "assistant", "Newest answer")
        assert _last_user_request(store, current_chat) == "Newest request"


def test_annotation_qc_followups_do_not_need_paths_repeated() -> None:
    assert _is_annotation_qc_followup("Proceed")
    assert _is_annotation_qc_followup("Do you think these blood clot polygons are correct?")
    assert not _is_annotation_qc_followup("Explain Leiden clustering")


def test_annotation_qc_bundle_is_recovered_only_from_the_open_chat() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        root = Path(temp_dir)
        store = AssistantStore(root / "assistant.sqlite3")
        current_chat = store.create_conversation()
        other_chat = store.create_conversation()
        source = root / "roi.ome.tiff"
        geojson = root / "clot.geojson"
        overview = root / "overview.png"
        tile = root / "tile.png"
        for path in (source, geojson, overview, tile):
            path.write_bytes(b"placeholder")
        common = {
            "conversation_id": current_chat,
            "request_message_id": "review-1",
            "source_path": str(source),
            "preview_mode": "qc_he",
            "channels": ["HE_Red", "HE_Green", "HE_Blue"],
            "overlay_sources": [{"path": str(geojson)}],
            "qc_summary": {"polygon_count": 1},
        }
        store.register_artifact(
            overview,
            kind="assistant_image_preview",
            metadata={**common, "qc_kind": "overview"},
        )
        store.register_artifact(
            tile,
            kind="assistant_image_preview",
            metadata={**common, "qc_kind": "candidate_tile", "candidate": 1},
        )
        bundle = _recent_annotation_qc_bundle(store, current_chat)
        assert bundle is not None
        assert bundle["request"]["source_path"] == str(source)
        assert [item["qc_kind"] for item in bundle["previews"]] == [
            "overview",
            "candidate_tile",
        ]
        assert _recent_annotation_qc_bundle(store, other_chat) is None


def test_channel_followup_reuses_the_latest_image_in_the_same_chat() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        root = Path(temp_dir)
        store = AssistantStore(root / "assistant.sqlite3")
        conversation = store.create_conversation()
        image = root / "registered.ome.tiff"
        preview = root / "preview.png"
        image.write_bytes(b"image placeholder")
        preview.write_bytes(b"preview placeholder")
        store.register_artifact(
            preview,
            kind="assistant_image_preview",
            metadata={"conversation_id": conversation, "source_path": str(image)},
        )
        assert _recent_preview_source(store, conversation) == str(image)


def test_visual_state_is_strictly_scoped_to_the_current_chat() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        root = Path(temp_dir)
        store = AssistantStore(root / "assistant.sqlite3")
        current_chat = store.create_conversation()
        other_chat = store.create_conversation()
        current_source = root / "current.ome.tiff"
        other_source = root / "other.ome.tiff"
        current_preview = root / "current.png"
        other_preview = root / "other.png"
        for path in (current_source, other_source, current_preview, other_preview):
            path.write_bytes(b"placeholder")
        store.register_artifact(
            current_preview,
            kind="assistant_image_preview",
            metadata={
                "conversation_id": current_chat,
                "request_message_id": "current-request",
                "source_path": str(current_source),
                "preview_mode": "markers",
                "channels": ["DAPI", "CD45", "CD68", "CD38"],
            },
        )
        store.register_artifact(
            other_preview,
            kind="assistant_image_preview",
            metadata={
                "conversation_id": other_chat,
                "request_message_id": "other-request",
                "source_path": str(other_source),
                "preview_mode": "he",
                "channels": ["HE_Red", "HE_Green", "HE_Blue"],
            },
        )
        states = _recent_preview_states(store, current_chat)
        assert len(states) == 1
        assert states[0]["source_path"] == str(current_source)
        assert states[0]["channels"] == ["DAPI", "CD45", "CD68", "CD38"]


def test_visual_state_discards_an_old_accidental_he_sibling() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        root = Path(temp_dir)
        store = AssistantStore(root / "assistant.sqlite3")
        conversation = store.create_conversation()
        source = root / "roi.ome.tiff"
        he_preview = root / "he.png"
        marker_preview = root / "markers.png"
        for path in (source, he_preview, marker_preview):
            path.write_bytes(b"placeholder")
        command = (
            "Display DAPI, CD45, CD68, and CD38 as a thumbnail. "
            "Add a scale bar to the image; each pixel is 0.5083 micrometer."
        )
        common = {
            "conversation_id": conversation,
            "request_message_id": "same-request",
            "source_path": str(source),
        }
        store.register_artifact(
            he_preview,
            kind="assistant_image_preview",
            source_command=command,
            metadata={**common, "preview_mode": "he", "channels": ["HE_Red", "HE_Green", "HE_Blue"]},
        )
        store.register_artifact(
            marker_preview,
            kind="assistant_image_preview",
            source_command=command,
            metadata={**common, "preview_mode": "markers", "channels": ["DAPI", "CD45", "CD68", "CD38"]},
        )
        states = _recent_preview_states(store, conversation)
        assert len(states) == 1
        assert states[0]["preview_mode"] == "markers"
        assert states[0]["channels"] == ["DAPI", "CD45", "CD68", "CD38"]


def test_empty_blur_trials_are_reported_as_no_result_and_no_visual_review() -> None:
    rows = [
        {
            "trial": f"trial_{index}",
            "polygon_count": 0,
            "separation_gap": 2.907,
        }
        for index in range(10)
    ]
    answer = _completed_blur_answer(
        "6b56fced-4139-4f46-b756-aae4f6d6d461",
        {
            "plan": {"run_output_dir": "/allowed/run"},
            "status": {"output_dir": "/allowed/run"},
            "report": {"result": {"comparison": rows, "selected_trial": "trial_0"}},
        },
    )
    assert "0 polygons" in answer
    assert "no valid best setting" in answer
    assert "No visual review was performed" in answer


def test_bright_glare_request_uses_registered_artifact_action() -> None:
    image = (
        "/data/spacemap/projects/study/outputs/rois/"
        "P11_roi_1.ome.tiff"
    )
    parsed = parse_artifact_optimization_request(
        "Investigate Bright glare and debris in the Annotation tab. "
        f"Test the best settings and assess the detected bright spots in {image}"
    )
    assert parsed is not None
    assert parsed["artifact_kind"] == "bright"
    assert parsed["image_path"] == image
    assert parsed["maximum_trials"] == 10
    assert parsed["output_parent"].endswith("/study/outputs/processed_annotations")
    assert parsed["tissue_boundary_path"].endswith(
        "/study/outputs/processed_annotations/P11_roi_1_tissue_boundary.geojson"
    )


def test_bright_followup_reuses_only_the_open_chat_request() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        current_chat = store.create_conversation()
        other_chat = store.create_conversation()
        wanted_image = (
            "/data/spacemap/projects/study/outputs/rois/"
            "wanted.ome.tiff"
        )
        wrong_image = (
            "/data/spacemap/projects/other/outputs/rois/"
            "wrong.ome.tiff"
        )
        store.add_message(
            current_chat,
            "user",
            f"Test the best Bright glare and debris settings for {wanted_image}",
        )
        store.add_message(
            other_chat,
            "user",
            f"Test the best Bright glare and debris settings for {wrong_image}",
        )
        followup = (
            "Use a brightfield-like channel and make a representative tile of each detected "
            "region so you can assess true and false positives."
        )
        store.add_message(current_chat, "user", followup)
        parsed = _artifact_draft_with_thread_context(followup, store, current_chat)
        assert parsed is not None
        assert parsed["image_path"] == wanted_image
        assert wrong_image not in parsed["user_request"]
