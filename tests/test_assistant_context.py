from spacemap_assistant.context import build_page_context


def test_page_context_includes_relevant_scalars_but_not_secrets_or_dataframes() -> None:
    state = {
        "active_project_name": "Example",
        "active_project_root": "/outside/project",
        "leiden_resolution": 0.8,
        "leiden_markers": ["CD3", "CD8"],
        "leiden_api_key": "never include",
        "leiden_large_object": {"many": "values"},
        "tab1_unrelated": "ignore",
    }
    context = build_page_context(
        "tab7",
        state,
        guide={"purpose": "Phenotype cells", "inputs": (), "outputs": ()},
    )
    assert context["page_title"] == "Phenotyping"
    assert context["assistant_can_open_project_files"] is False
    assert context["current_settings"]["leiden_resolution"] == 0.8
    assert context["current_settings"]["leiden_markers"] == ["CD3", "CD8"]
    assert "leiden_api_key" not in context["current_settings"]
    assert "leiden_large_object" not in context["current_settings"]
    assert "tab1_unrelated" not in context["current_settings"]


def test_page_context_includes_evidence_published_by_the_current_tab() -> None:
    state = {
        "_spacemap_ai_page_evidence": {
            "tab4": {
                "section": "pixel_classification",
                "sample_id": "P11_roi_1",
                "assignment_csv": "/allowed/step3_cluster_assignments.csv",
                "profile_tables": {"shrunk_z": "/allowed/step3_cluster_shrunk_z_values.csv"},
            }
        }
    }
    context = build_page_context("tab4", state)
    assert context["current_tab_evidence"]["section"] == "pixel_classification"
    assert context["current_tab_evidence"]["sample_id"] == "P11_roi_1"
    assert context["current_tab_evidence"]["profile_tables"]["shrunk_z"].endswith(
        "step3_cluster_shrunk_z_values.csv"
    )
