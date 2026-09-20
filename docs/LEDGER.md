# Change ledger

Every change ever made to this repository, newest first: what changed, who made
it, and why.

This file is generated from the Git history by `scripts/update_ledger.py`. Do
not edit it by hand -- run the script instead, or let CI regenerate it. The
"Why" column comes from the `Why:` line in each commit message; see
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to write one.

**11 changes** by **1 contributor**, touching **290 files**. Latest: `1784367` on 2026-09-19.


## 2026-09

| Date | Change | Who | Why | Files | Lines |
| --- | --- | --- | --- | --- | --- |
| 2026-09-19 | `1784367` docs: add ORCID identifiers to the citation metadata | Bassel Dawod | an ORCID is what lets indexers and reference managers attribute the software to the right people when names are ambiguous or change. | 1 | +6/-3 |
| 2026-09-19 | `3291334` docs: record UTSW copyright and named authorship | Bassel Dawod | copyright ownership, software authorship and paper authorship are three different things, and a single lab name in every slot recorded none of them correctly. | 3 | +74/-9 |
| 2026-09-19 | `47d7c92` chore: drop the empty runtime/ directory | Bassel Dawod | it was a leftover from the internal checkout and only invited the question of what it was for. | 3 | +0/-22 |
| 2026-09-19 | `59c7b5d` fix: repair three latent NameError crashes found by static analysis | Bassel Dawod | all three were reachable crashes in released analysis code, and the batch-correction one meant a QC figure users believe they are getting was never produced. | 4 | +3285/-3284 |
| 2026-09-19 | `8803cce` fix: make the project installable, and stop the container forcing the AI on | Bassel Dawod | every route into the project began with an install that could not succeed, and in the container the assistant was not actually optional. | 8 | +392/-14 |
| 2026-09-19 | `8964ae3` fix: keep the ledger stable when a commit only rewrites it | Bassel Dawod | without this the ledger was permanently one commit behind itself and the CI check could never pass. | 2 | +21/-2 |
| 2026-09-19 | `2fd931d` docs: generate the change ledger | Bassel Dawod | Git already holds the history, but reading it requires knowing Git; the ledger puts the same record in front of a reviewer, an auditor or a new lab member as a single page. | 2 | +348/-3 |
| 2026-09-19 | `24f2903` ci: add licensing, contribution workflow and automated repository checks | Bassel Dawod | the change ledger is only as good as the commit messages behind it, and the no-extras and path checks stop the two regressions that would most easily slip back in. | 12 | +819/-0 |
| 2026-09-19 | `2833155` docs: write installation, assistant, architecture and troubleshooting guides | Bassel Dawod | the previous notes recorded internal deliberation and site-specific operating procedure, neither of which helps an external user install or run the pipeline. | 6 | +1044/-0 |
| 2026-09-19 | `7b34da7` build: package SPACEMAP for installation by other groups | Bassel Dawod | a new user must be able to install and start SPACEMAP without reproducing one machine's environment by hand. | 8 | +473/-0 |
| 2026-09-19 | `288f52d` feat: import the SPACEMAP pipeline, GUI and test suite | Bassel Dawod | this is the first public release of a pipeline that until now only ran from one lab's checkout on one cluster. | 262 | +230730/-0 |

---

## Files changed per commit

<details><summary><code>1784367</code> docs: add ORCID identifiers to the citation metadata</summary>

- `CITATION.cff`

</details>

<details><summary><code>3291334</code> docs: record UTSW copyright and named authorship</summary>

- `CITATION.cff`
- `LICENSE`
- `pyproject.toml`

</details>

<details><summary><code>47d7c92</code> chore: drop the empty runtime/ directory</summary>

- `.dockerignore`
- `.gitignore`
- `runtime/README.md`

</details>

<details><summary><code>59c7b5d</code> fix: repair three latent NameError crashes found by static analysis</summary>

- `pyproject.toml`
- `src/spacemap/batch_correction_utils.py`
- `src/spacemap/segmentation_cellpose.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`

</details>

<details><summary><code>8803cce</code> fix: make the project installable, and stop the container forcing the AI on</summary>

- `Dockerfile`
- `docker-compose.yml`
- `pyproject.toml`
- `requirements-lock.txt`
- `requirements.txt`
- `src/spacemap/ai_settings.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `tests/test_f1_paper_figures.py`

</details>

<details><summary><code>8964ae3</code> fix: keep the ledger stable when a commit only rewrites it</summary>

- `docs/LEDGER.md`
- `scripts/update_ledger.py`

</details>

<details><summary><code>2fd931d</code> docs: generate the change ledger</summary>

- `docs/LEDGER.md`
- `scripts/update_ledger.py`

</details>

<details><summary><code>24f2903</code> ci: add licensing, contribution workflow and automated repository checks</summary>

- `.githooks/commit-msg`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/workflows/ci.yml`
- `.gitmessage`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `LICENSE`
- `scripts/check_no_local_paths.py`
- `scripts/install_hooks.sh`
- `scripts/update_ledger.py`

</details>

<details><summary><code>2833155</code> docs: write installation, assistant, architecture and troubleshooting guides</summary>

- `CHANGELOG.md`
- `README.md`
- `docs/ai-assistant.md`
- `docs/architecture.md`
- `docs/installation.md`
- `docs/troubleshooting.md`

</details>

<details><summary><code>7b34da7</code> build: package SPACEMAP for installation by other groups</summary>

- `.dockerignore`
- `Dockerfile`
- `docker-compose.yml`
- `environment.yml`
- `pyproject.toml`
- `requirements.txt`
- `run_interface.sh`
- `runtime/README.md`

</details>

<details><summary><code>288f52d</code> feat: import the SPACEMAP pipeline, GUI and test suite</summary>

- `.gitignore`
- `marker_configs.json`
- `notebooks/Step1.1_registration_codex_codex.ipynb`
- `notebooks/Step10A_spatial_analysis.ipynb`
- `notebooks/Step10_sub_phenotypes.ipynb`
- `notebooks/Step11_spatial_beds.ipynb`
- `notebooks/Step12_spatial_figures_v4_tumor_clones.ipynb`
- `notebooks/Step12_spatial_figures_v5_cells.ipynb`
- `notebooks/Step1A_background_subtraction.ipynb`
- `notebooks/Step1_registration.ipynb`
- `notebooks/Step2B_background_detection.ipynb`
- `notebooks/Step2_roi_extraction.ipynb`
- `notebooks/Step3A_segmentation_cellpose.ipynb`
- `notebooks/Step3B_segmentation_deepcell.ipynb`
- `notebooks/Step4B_cell_annotation.ipynb`
- `notebooks/Step4_annotation.ipynb`
- `notebooks/Step5_batch_correction.ipynb`
- `notebooks/Step6A_phenotype_leiden.ipynb`
- `notebooks/Step6A_phenotype_leiden_subclustering.ipynb`
- `notebooks/Step6B_phenotype_SOM.ipynb`
- `notebooks/Step6C_phenotype_threshold.ipynb`
- `notebooks/Step6D_phenotype_SCIMAP.ipynb`
- `notebooks/Step6E_phenotype_vortex.ipynb`
- `notebooks/Step7_accessories.ipynb`
- `notebooks/Step8_Comparison.ipynb`
- `notebooks/Step9A_ML_manual_reference.ipynb`
- `notebooks/Step9B_ML_concensus.ipynb`
- `notebooks/new/tissue_annotation_v2.ipynb`
- `src/spacemap/__init__.py`
- `src/spacemap/acc_utils.py`
- `src/spacemap/ai_settings.py`
- `src/spacemap/annotation_utils.py`
- `src/spacemap/background_detection.py`
- `src/spacemap/background_subtraction_utils.py`
- `src/spacemap/batch_correction_utils.py`
- `src/spacemap/cell_spatial_annotation.py`
- `src/spacemap/comparison_utils.py`
- `src/spacemap/config.py`
- `src/spacemap/grouped_validation.py`
- `src/spacemap/hardware.py`
- `src/spacemap/ml_consensus.py`
- `src/spacemap/ml_mr_utils.py`
- `src/spacemap/model_catalog.json`
- `src/spacemap/model_catalog.py`
- `src/spacemap/ph_accel_utils.py`
- `src/spacemap/ph_leiden_utils.py`
- `src/spacemap/ph_scimap_utils.py`
- `src/spacemap/ph_som_utils.py`
- `src/spacemap/ph_subpheno_utils.py`
- `src/spacemap/ph_thres_utils.py`
- `src/spacemap/ph_vortex_utils.py`
- `src/spacemap/registration.py`
- `src/spacemap/segmentation_cellpose.py`
- `src/spacemap/segmentation_deepcell.py`
- `src/spacemap/spatial_analysis_utils.py`
- `src/spacemap/spatial_beds_utils.py`
- `src/spacemap/spatial_distance_acceleration.py`
- `src/spacemap/spatial_visibility.py`
- `src/spacemap/tissue_annotation_figures.py`
- `src/spacemap/tissue_annotation_utils.py`
- `src/spacemap/tissue_continuity.py`
- `src/spacemap/userpaths.py`
- `src/spacemap/utils.py`
- `src/spacemap_assistant/__init__.py`
- `src/spacemap_assistant/action_catalog.py`
- `src/spacemap_assistant/action_monitor.py`
- `src/spacemap_assistant/action_notebook.py`
- `src/spacemap_assistant/blur_action.py`
- `src/spacemap_assistant/blur_worker.py`
- `src/spacemap_assistant/cluster.py`
- `src/spacemap_assistant/cluster_labeling.py`
- `src/spacemap_assistant/code_index.py`
- `src/spacemap_assistant/context.py`
- `src/spacemap_assistant/general_action.py`
- `src/spacemap_assistant/general_worker.py`
- `src/spacemap_assistant/instructions.py`
- `src/spacemap_assistant/memory.py`
- `src/spacemap_assistant/model_client.py`
- `src/spacemap_assistant/model_installer.py`
- `src/spacemap_assistant/model_runtime.py`
- `src/spacemap_assistant/model_service/README.md`
- `src/spacemap_assistant/model_service/__init__.py`
- `src/spacemap_assistant/model_service/server.py`
- `src/spacemap_assistant/online_access.py`
- `src/spacemap_assistant/paths.py`
- `src/spacemap_assistant/planner_paths.py`
- `src/spacemap_assistant/preview_action.py`
- `src/spacemap_assistant/registration_action.py`
- `src/spacemap_assistant/registration_worker.py`
- `src/spacemap_assistant/roi_action.py`
- `src/spacemap_assistant/roi_worker.py`
- `src/spacemap_assistant/storage.py`
- `src/spacemap_gui/__init__.py`
- `src/spacemap_gui/app.py`
- `src/spacemap_gui/cache_manager.py`
- `src/spacemap_gui/cell_visualizer.py`
- `src/spacemap_gui/components/__init__.py`
- `src/spacemap_gui/components/assistant_actions.py`
- `src/spacemap_gui/components/assistant_panel.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `src/spacemap_gui/file_browser.py`
- `src/spacemap_gui/interface_config.json`
- `src/spacemap_gui/launch.py`
- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/process_log.py`
- `src/spacemap_gui/project_bundle.py`
- `src/spacemap_gui/requirements.txt`
- `src/spacemap_gui/resolve_rule_parser.py`
- `src/spacemap_gui/tabs/__init__.py`
- `src/spacemap_gui/tabs/cell_tagging_ui.py`
- `src/spacemap_gui/tabs/legacy_cell_annotation.py`
- `src/spacemap_gui/tabs/segmentation_spatial_ui.py`
- `src/spacemap_gui/tabs/tab0_project.py`
- `src/spacemap_gui/tabs/tab1_registration.py`
- `src/spacemap_gui/tabs/tab2_background_detection.py`
- `src/spacemap_gui/tabs/tab2_background_subtraction.py`
- `src/spacemap_gui/tabs/tab3_roi_extraction.py`
- `src/spacemap_gui/tabs/tab4_annotation/__init__.py`
- `src/spacemap_gui/tabs/tab4_annotation/common.py`
- `src/spacemap_gui/tabs/tab4_annotation/figures.py`
- `src/spacemap_gui/tabs/tab4_annotation/heatmap_component.py`
- `src/spacemap_gui/tabs/tab4_annotation/s1_tissue_detection.py`
- `src/spacemap_gui/tabs/tab4_annotation/s2_artifact_removal.py`
- `src/spacemap_gui/tabs/tab4_annotation/s2b_tissue_pieces.py`
- `src/spacemap_gui/tabs/tab4_annotation/s3_pixel_classification.py`
- `src/spacemap_gui/tabs/tab4_annotation/s4_pool_and_zones.py`
- `src/spacemap_gui/tabs/tab5_segmentation.py`
- `src/spacemap_gui/tabs/tab6_batch_correction.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/__init__.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/cell_visualization.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/common.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/consensus_ml.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/__init__.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/index.html`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/package-lock.json`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/package.json`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/src/index.js`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/src/index.ts`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/src/vite-env.d.ts`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/tsconfig.json`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/vite.config.ts`
- `src/spacemap_gui/tabs/tab7_phenotyping/leiden_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/marker_composition.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/mr_ml_phenotyping.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/resolve_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/scimap_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/som_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/subphenotyping.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/threshold_gating.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/visualization.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/vortex_clustering.py`
- `src/spacemap_gui/tabs/tab8_analysis/__init__.py`
- `src/spacemap_gui/tabs/tab8_analysis/_dens_copy_test.py`
- `src/spacemap_gui/tabs/tab8_analysis/actual_tissue.py`
- `src/spacemap_gui/tabs/tab8_analysis/cell_cell_interaction.py`
- `src/spacemap_gui/tabs/tab8_analysis/clonality.py`
- `src/spacemap_gui/tabs/tab8_analysis/common.py`
- `src/spacemap_gui/tabs/tab8_analysis/density_analysis.py`
- `src/spacemap_gui/tabs/tab8_analysis/density_clouds.py`
- `src/spacemap_gui/tabs/tab8_analysis/feature_screening.py`
- `src/spacemap_gui/tabs/tab8_analysis/feature_screening_core.py`
- `src/spacemap_gui/tabs/tab8_analysis/neighborhoods.py`
- `src/spacemap_gui/tabs/tab8_analysis/roi_explorer.py`
- `src/spacemap_gui/tabs/tab8_analysis/roi_explorer_core.py`
- `src/spacemap_gui/tabs/tab8_analysis/triads.py`
- `src/spacemap_gui/tabs/tab8_analysis/virtual_tissue.py`
- `src/spacemap_gui/tabs/tab9_visualization/__init__.py`
- `src/spacemap_gui/tabs/tab9_visualization/actual_tissue.py`
- `src/spacemap_gui/tabs/tab9_visualization/cell_visualization.py`
- `src/spacemap_gui/tabs/tab9_visualization/common.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_paper_analysis.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_paper_figures.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_strict_runner.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_strict_validation_ui.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_validation_ui.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_validation_utils.py`
- `src/spacemap_gui/tabs/tab9_visualization/heatmap.py`
- `src/spacemap_gui/tabs/tab9_visualization/heatmap_utils.py`
- `src/spacemap_gui/tabs/tab9_visualization/marker_analysis.py`
- `src/spacemap_gui/tabs/tab9_visualization/overlay_core.py`
- `src/spacemap_gui/tabs/tab9_visualization/pairwise.py`
- `src/spacemap_gui/tabs/tab9_visualization/phenotype_colors.py`
- `src/spacemap_gui/tabs/tab9_visualization/proportion.py`
- `src/spacemap_gui/tabs/tab9_visualization/umap_explorer.py`
- `src/spacemap_gui/tabs/tab9_visualization/virtual_tissue.py`
- `src/spacemap_gui/tabs/tab9_visualization/visual_validation.py`
- `src/spacemap_gui/tabs/tab_ai_assistant.py`
- `src/spacemap_gui/ui_utils.py`
- `src/spacemap_gui/workflow_guidance.py`
- `src/spacemap_workflows/__init__.py`
- `src/spacemap_workflows/annotation.py`
- `src/spacemap_workflows/leiden_job.py`
- `src/spacemap_workflows/leiden_pipeline.py`
- `src/spacemap_workflows/registry.py`
- `src/spacemap_workflows/schemas.py`
- `tests/conftest.py`
- `tests/test_action_notebook.py`
- `tests/test_annotation_button_state.py`
- `tests/test_annotation_output_layout.py`
- `tests/test_annotation_single_source.py`
- `tests/test_artifact_blur_detection.py`
- `tests/test_assistant_chat_context.py`
- `tests/test_assistant_cluster.py`
- `tests/test_assistant_context.py`
- `tests/test_assistant_general_action.py`
- `tests/test_assistant_instructions.py`
- `tests/test_assistant_memory.py`
- `tests/test_assistant_model_client.py`
- `tests/test_assistant_model_proxy.py`
- `tests/test_assistant_online_access.py`
- `tests/test_assistant_paths.py`
- `tests/test_assistant_preview_action.py`
- `tests/test_assistant_registration_action.py`
- `tests/test_assistant_roi_action.py`
- `tests/test_assistant_storage.py`
- `tests/test_assistant_ui_smoke.py`
- `tests/test_background_detection.py`
- `tests/test_background_image_writer.py`
- `tests/test_blood_clot_detection.py`
- `tests/test_blur_parameter_search.py`
- `tests/test_cache_manager.py`
- `tests/test_cell_overlay_coordinates.py`
- `tests/test_cell_spatial_annotation.py`
- `tests/test_cell_tagging_ui.py`
- `tests/test_cluster_labeling.py`
- `tests/test_f1_paper_figures.py`
- `tests/test_f1_validation_method_colors.py`
- `tests/test_generic_channel_fallback.py`
- `tests/test_leiden_background_job.py`
- `tests/test_manual_reference_clear_label.py`
- `tests/test_manual_reference_fresh_source.py`
- `tests/test_manual_reference_get_more_cells.py`
- `tests/test_manual_reference_pending_assignments.py`
- `tests/test_manual_reference_top_markers.py`
- `tests/test_marker_composition.py`
- `tests/test_model_service.py`
- `tests/test_persistence_action_keys.py`
- `tests/test_phenotype_geojson_tagging.py`
- `tests/test_phenotype_recommendation_rules.py`
- `tests/test_preview_cache_keys.py`
- `tests/test_project_bundle.py`
- `tests/test_project_isolation.py`
- `tests/test_registration_pyramid_output.py`
- `tests/test_resolve_rule_parser.py`
- `tests/test_resolve_threshold_loading.py`
- `tests/test_resolve_transform_pipeline.py`
- `tests/test_roi_label_resolution.py`
- `tests/test_spacemap_tool_registry.py`
- `tests/test_spatial_distance_acceleration.py`
- `tests/test_spatial_distance_search.py`
- `tests/test_spatial_visibility.py`
- `tests/test_subphenotyping_utils.py`
- `tests/test_tab1_intermediate_subdir.py`
- `tests/test_tab1_widget_defaults.py`
- `tests/test_tab3_pair_settings.py`
- `tests/test_tab3_roi_table.py`
- `tests/test_tab8_publication_stats.py`
- `tests/test_tab9_phenotype_colors.py`
- `tests/test_threshold_directory_workspace.py`
- `tests/test_tissue_continuity.py`
- `tests/test_widget_range_guard.py`

</details>

