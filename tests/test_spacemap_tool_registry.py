from pathlib import Path

from spacemap_assistant.blur_action import parse_blur_optimization_request
from spacemap_assistant.code_index import source_context_for_prompt
from spacemap_assistant.general_action import _registered_tools_in_code
from spacemap_assistant import planner_paths
from spacemap_workflows.registry import get_tool, relevant_tools, tool_counts


def test_full_backend_catalog_contains_every_pipeline_area() -> None:
    counts = tool_counts()
    assert counts["total"] > 500
    for page in (
        "1. Registration",
        "2. Background",
        "3. ROI",
        "4. Annotation",
        "5. Segmentation",
        "6. Batch correction",
        "7. Phenotyping",
        "8. Spatial analysis",
        "9. Visualization",
    ):
        assert counts["pages"][page] > 0


def test_known_backend_entry_points_are_registered() -> None:
    expected = (
        "spacemap.registration.process_single_image_rois",
        "spacemap.background_detection.detect_backgrounds",
        "spacemap.segmentation_cellpose.segment_image_cellpose",
        "spacemap.batch_correction_utils.BatchCorrector.correct_batch_effects",
        "spacemap.ph_leiden_utils.leiden_hpo",
        "spacemap.spatial_beds_utils.run_spatial_beds_analysis",
    )
    for name in expected:
        assert get_tool(name).callable_path


def test_retrieval_keeps_named_algorithm_in_the_relevant_group() -> None:
    leiden = relevant_tools(
        "run Leiden clustering and compare quality to a manual reference",
        limit=12,
    )
    cellpose = relevant_tools("segment cells with Cellpose", limit=8)
    assert any(item.callable_path == "spacemap.ph_leiden_utils:leiden_hpo" for item in leiden)
    assert any(
        item.callable_path == "spacemap.segmentation_cellpose:segment_image_cellpose"
        for item in cellpose
    )


def test_long_request_gets_tools_from_every_named_pipeline_stage() -> None:
    tools = relevant_tools(
        "register CODEX and H&E, remove background, extract ROIs, annotate valid tissue, "
        "segment cells, correct batches, phenotype clusters, run spatial analysis, and make a heatmap",
        limit=18,
    )
    pages = {item.page for item in tools}
    assert {
        "1. Registration",
        "2. Background",
        "3. ROI",
        "4. Annotation",
        "5. Segmentation",
        "6. Batch correction",
        "7. Phenotyping",
        "8. Spatial analysis",
        "9. Visualization",
    } <= pages


def test_source_evidence_contains_only_registered_public_entry_points() -> None:
    context = source_context_for_prompt(
        "test Leiden clustering against a manual reference and save a heatmap"
    )
    assert "CALLABLE: spacemap.ph_leiden_utils:" in context
    assert "CALLABLE: spacemap_gui.tabs.tab7_phenotyping.tools:_" not in context
    assert "PhenotypeAssignmentApp." not in context


def test_generated_job_records_the_registered_function() -> None:
    code = """
from spacemap.ph_leiden_utils import leiden_hpo

result = leiden_hpo(None)
OUTPUT_DIR.joinpath("result.txt").write_text(str(result))
"""
    assert _registered_tools_in_code(code) == ["spacemap.ph_leiden_utils.leiden_hpo"]


def test_generated_job_records_a_registered_backend_class_method() -> None:
    code = """
from spacemap.batch_correction_utils import BatchCorrector

corrector = BatchCorrector(method='combat')
result = corrector.correct_batch_effects(None)
OUTPUT_DIR.joinpath("result.txt").write_text(str(result))
"""
    assert _registered_tools_in_code(code) == [
        "spacemap.batch_correction_utils.BatchCorrector.correct_batch_effects"
    ]


def test_online_planner_uses_local_aliases_not_directory_names(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "private_study" / "sample.ome.tiff"
    image.parent.mkdir()
    image.write_bytes(b"test")
    output = tmp_path / "private_study" / "outputs" / "segmentation"
    monkeypatch.setattr(
        planner_paths,
        "require_assistant_read_path",
        lambda path, must_exist=False: Path(path).resolve(),
    )
    monkeypatch.setattr(
        planner_paths,
        "require_assistant_write_path",
        lambda path: Path(path).resolve(),
    )
    prompt = f"Segment {image} and save results in {output}"
    sanitized, aliases = planner_paths.online_action_content(prompt)
    assert str(tmp_path) not in sanitized
    assert "LOCAL_PATH_1" in sanitized
    restored = planner_paths.restore_online_job_paths(
        {
            "inputs": {"image": "LOCAL_PATH_1"},
            "output_dir": "LOCAL_PATH_2",
            "parameters": {"source": "LOCAL_PATH_1"},
        },
        aliases,
    )
    assert restored["inputs"]["image"] == str(image.resolve())
    assert restored["output_dir"] == str(output.resolve())


def test_blur_request_builds_an_experiment_draft_without_preview_routing() -> None:
    root = Path("/data/spacemap/projects/demo")
    prompt = (
        f"Test the best blur parameters for {root / 'outputs/rois/P1.ome.tiff'} and save "
        f"all trials in {root / 'outputs/processed_annotations/test'}"
    )
    draft = parse_blur_optimization_request(prompt)
    assert draft is not None
    assert draft["maximum_trials"] == 10
    assert draft["output_parent"].endswith("/outputs/processed_annotations/test")
