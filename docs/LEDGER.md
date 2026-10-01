# Change ledger

Every change ever made to this repository, newest first: what changed, who made
it, and why.

This file is generated from the Git history by `scripts/update_ledger.py`. Do
not edit it by hand -- run the script after committing changes. The
"Why" column comes from the `Why:` line in each commit message; see
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to write one.

Local machine paths, account identifiers and site-specific cluster names are
redacted in this public rendering. The original Git history is unchanged.

**48 changes** by **1 contributor**, touching **344 files**. Latest: `7405eec` on 2026-10-01.


## 2026-10

| Date | Change | Who | Why | Files | Lines |
| --- | --- | --- | --- | --- | --- |
| 2026-10-01 | `7405eec` fix: keep local infrastructure details out of publication files | Bassel Dawod | generated documentation and examples should not reintroduce machine paths from historical commits. | 7 | +45/-13 |
| 2026-10-01 | `b557e38` fix: prepare SPACEMAP installation and packaging for publication | Bassel Dawod | a new user could encounter unsupported dependencies, missing GUI assets, misplaced session state and unclear installation instructions; the release also needs consistent institutional licensing and no personal account paths. | 48 | +5409/-964 |

## 2026-09

| Date | Change | Who | Why | Files | Lines |
| --- | --- | --- | --- | --- | --- |
| 2026-09-29 | `6ab5193` feat(rois): a crop can keep its whole box instead of blanking outside the shape | Bassel Dawod | some analyses need the full rectangle around an ROI, not only the pixels inside its outline. | 2 | +27/-2 |
| 2026-09-28 | `1fee954` feat(images): ROI crops and background-removed images keep their parent's metadata | Bassel Dawod | the crops are the images every later step uses; they must carry the same acquisition details as the registered image. | 7 | +280/-6 |
| 2026-09-28 | `aa99c1f` refactor(tab5): segmentation region and cell tagging run from saved settings; annotation chosen by hand | Bassel Dawod | Studio's Segmentation and Cell tagging must run exactly as Tab 5 does, and mismatched names had no fix. | 6 | +210/-94 |
| 2026-09-28 | `095d2bf` feat(tab1): any CODEX image can be paired with any H&E image by hand | Bassel Dawod | with names that do not match (S7 HE scan.tif against slide7.qptiff) a pair could not be made at all. | 4 | +127/-30 |
| 2026-09-28 | `272039a` feat(registration): the merged image keeps all of the CODEX metadata | Bassel Dawod | the registered image is the project's source image; losing the acquisition metadata (exposure, filters, slide) there loses it for every later step. | 7 | +431/-12 |
| 2026-09-28 | `7717de7` perf(tab8): cell-cell interactions count pairs with bincount | Bassel Dawod | Studio runs this test on whole projects (millions of cells, 500 permutations), where np.add.at took hours. | 3 | +53/-9 |
| 2026-09-28 | `c26d11a` feat(tab6): cyCombine's own batch correction, exact, as the default | Bassel Dawod | the user asked for the exact cyCombine approach; the earlier translation used another SOM and ComBat library, a size cut-off and a global clip, and the default (monotonic) widened each batch. | 6 | +665/-34 |
| 2026-09-27 | `fe018b2` fix(assistant): "stop the model with SPACEMAP" can be saved | Bassel Dawod | Studio's AI setup offers the choice; turning it off did nothing. | 2 | +11/-0 |
| 2026-09-27 | `20ad9a5` fix(assistant): the local model server on a cluster: one record per node, no leftovers, a free port | Bassel Dawod | the user saw "Port 8091 is already in use" and asked to stop servers left running and to have the model stop with the session. | 3 | +339/-24 |
| 2026-09-25 | `4de20b9` feat(assistant): one AI setup page; the chosen model starts when you ask | Bassel Dawod | setting up the assistant meant going between Tab 0 (packages, model library, token), the assistant page's Model & Compute tab (provider and API key, start and stop the model, cluster), its Settings tab and Admin Access. A downloaded model still had to be "selected" in Tab 0 before Start worked, the GPU memory menu sent names the runtime rejects (only Auto worked), and nothing could be added without typing a Hugging Face repository by hand. | 9 | +867/-390 |
| 2026-09-25 | `0f1b0c1` build: one environment for SPACEMAP, the viewer and the assistant; spacemap doctor | Bassel Dawod | the cluster environment had drifted: an old editable SPACEMAP record (V16, its path file pointing at V17) made pip print "requires deepcell / adjusttext" on every install, both OpenCV packages, scikit-image newer than palom allows, and the assistant's packages added by hand (transformers 5 then broke the model server). A fresh install could not even resolve: palom >= 2025 needs zarr >= 2.17, which scimap (Tab 7's SCIMAP method) forbids. | 8 | +492/-173 |
| 2026-09-25 | `79c406b` fix: Tab 3 reads the last step's images; [local directory] and [local directory] are one folder | Bassel Dawod | in PDAC (and INNATE_all) Tab 3's input had become its own ROI output folder, and Tab 2's output with it, because the two were tied by a folder link. And a project opened as [local directory] but saved as [local directory] (the same folder on the cluster) showed every folder as changed, so pages saved the project and updated again by themselves. | 4 | +255/-8 |
| 2026-09-25 | `4922cff` fix(assistant): the local model answers; live download progress | Bassel Dawod | every question to the local model failed with "Only google/gemma-4-12B-it is installed". The runtime starts the model server under the model's Hugging Face name, but the chat sent SPACEMAP's short id, which the server refuses. The chat now sends the same name the runtime uses. | 5 | +147/-15 |
| 2026-09-25 | `7ea0ff8` fix: keep SPACEMAP's settings out of the node-local /tmp folder | Bassel Dawod | UTSW the cluster sets XDG_DATA_HOME=/tmp/$USER for every user (for podman), and SPACEMAP put its home there. /tmp is local to one node and cleaned, so the assistant's settings and downloaded models, the viewer's state and the interface cache differed from node to node or vanished (a 24 GB model went to one node's /tmp; old folder choices came back from another node's cache). | 3 | +98/-4 |
| 2026-09-25 | `330c0a9` fix(assistant): keep Python's own itertools.pairwise so the model server loads | Bassel Dawod | spacemap/utils.py replaced itertools.pairwise with its own function (a workaround for palom on Python 3.9). With transformers 5, loading a model imports torch's compiler, which rejects the replacement, so the assistant's model server failed at import ("substitute_in_graph expects a class"). The replacement is now made only where Python lacks pairwise; SPACEMAP requires Python 3.10+, which has it. | 2 | +36/-2 |
| 2026-09-25 | `2a8050b` fix(assistant): clear instructions for installing what the local model needs | Bassel Dawod | the user could not tell which environment to install into, whether a separate one was needed, or what was already installed. | 5 | +172/-23 |
| 2026-09-25 | `6371845` test: every page must still render (guard for changes elsewhere) | Bassel Dawod | the image viewer work must leave every SPACEMAP feature intact, and this makes a broken page fail the tests. | 2 | +153/-0 |
| 2026-09-25 | `afd8ad2` fix(tab3): refill blank marker names and black colours from the image | Bassel Dawod | the P11 channel list showed no names and black colours. | 3 | +84/-10 |
| 2026-09-25 | `aa5a947` fix: run log crashed on settings keyed by numbers | Bassel Dawod | clicking an image in Tab 3 of the INNATE project raised AttributeError. | 2 | +34/-3 |
| 2026-09-25 | `c0b12ec` fix: switching to a moved project no longer loops forever | Bassel Dawod | switching from PDAC to INNATE_all_083120206 (moved from SPACEMAP_V17 to SPACEMAP_github/project) froze the interface. | 5 | +117/-6 |
| 2026-09-24 | `cb65845` fix: keep data out of the project file, and shrink files it already bloated | Bassel Dawod | the PDAC project file reached 44 GB and SPACEMAP hung at start-up using over 60 GB of memory; this had happened before with other results. | 4 | +491/-6 |
| 2026-09-22 | `dcd8d79` feat: one run log per tab, and phenotype saves without rewriting the CSV | Bassel Dawod | the user found Leiden's log box off and asked for one consistently named, always-on log that records settings changes, time and compute; and asked for fast Parquet-backed saves that survive an interface crash. | 24 | +1893/-167 |
| 2026-09-22 | `1cd78dd` fix: save every setting, unify Tab 7 layout and terminal output | Bassel Dawod | preparing SPACEMAP for publication. Users lost settings on reload, Tab 7 methods looked and read differently, terminal output could not be followed, and two notebooks silently did not run what they describe. | 67 | +1746/-1270 |
| 2026-09-22 | `324de8b` fix: allowlist the docs that describe the path rule | Bassel Dawod | a check that fires on its own documentation trains people to ignore it. | 1 | +5/-0 |
| 2026-09-22 | `252ad0e` docs: add a handover document and a publishing procedure | Bassel Dawod | everything here was learned once, at cost, and lived only in one conversation. Someone starting tomorrow would have rediscovered all of it. | 4 | +551/-0 |
| 2026-09-22 | `d7cc4eb` feat: rank a cell's top markers by percentile, not normalized score | Bassel Dawod | the top five markers steer a human's phenotype call, so a ranking that systematically favours rare markers biases the reference set that downstream models are trained and validated against. | 1 | +94/-8 |
| 2026-09-22 | `11ed7ce` fix: recognise button keys that begin and end with a placeholder | Bassel Dawod | a saved project could make the tab unopenable, and the cause was invisible because the widget key never appeared in any list the scanner produced. | 1 | +42/-4 |
| 2026-09-22 | `2426062` fix: move the label commit control out of the form, and stop project data being committed | Bassel Dawod | the feature was unreachable, and the repository was one push away from publishing analysis project files. | 6 | +211/-6 |
| 2026-09-22 | `f574f68` docs: add a guide for using this version on real data | Bassel Dawod | someone adopting this version needs to know their data cannot leak and that a saved label is not yet in the CSV, and neither was written down. | 2 | +259/-0 |
| 2026-09-22 | `01ec57f` fix: name the two save actions for what they actually do | Bassel Dawod | the previous wording described the old behaviour and quietly misled anyone relying on it. | 1 | +4/-4 |
| 2026-09-22 | `cc428ef` fix: put the commit control where labels are saved, and say where they went | Bassel Dawod | a user finished a labelling session, was told the save was "completed and verified", and had no way to tell that the CSV on disk was unchanged or which control would change it. | 1 | +75/-2 |
| 2026-09-22 | `0110154` perf: cache the segmentation index, and store labels without rewriting the table | Bassel Dawod | labelling fifty cells took around thirteen minutes of waiting, and two earlier attempts optimised paths that were not the bottleneck because nothing reported where the time was going. | 8 | +777/-95 |
| 2026-09-22 | `64a4808` fix: correct loader dtypes, and normalize only the cells shown | Bassel Dawod | a reviewer waited thirteen minutes to label fifty cells, and the screen errored when they tried. Both costs were avoidable work rather than anything inherent to the analysis. | 3 | +235/-2 |
| 2026-09-22 | `dd094d7` perf: read only the columns a screen needs, from a columnar copy | Bassel Dawod | the pipeline became unusable on real cohort-sized data, and the cost was entirely avoidable work rather than anything inherent to the analysis. | 21 | +1268/-145 |
| 2026-09-19 | `7335961` docs: record the submitted manuscript's author list | Bassel Dawod | capturing the order now means the citation is a two-line edit on acceptance rather than a reconstruction from memory. | 1 | +71/-0 |
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

<details><summary><code>7405eec</code> fix: keep local infrastructure details out of publication files</summary>

- `docs/handover.md`
- `scripts/check_no_local_paths.py`
- `scripts/update_ledger.py`
- `src/spacemap_gui/components/assistant_panel.py`
- `tests/test_assistant_ui_smoke.py`
- `tests/test_pipeline_folders.py`
- `tests/test_public_ledger.py`

</details>

<details><summary><code>b557e38</code> fix: prepare SPACEMAP installation and packaging for publication</summary>

- `.dockerignore`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/workflows/ci.yml`
- `.gitignore`
- `CHANGELOG.md`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `Dockerfile`
- `LICENSE`
- `README.md`
- `docs/first-project.md`
- `docs/handover.md`
- `docs/installation.md`
- `docs/publishing-changes.md`
- `docs/working-with-this-repository.md`
- `environment.yml`
- `pyproject.toml`
- `requirements-lock.txt`
- `scripts/check_installed_app.py`
- `scripts/check_no_local_paths.py`
- `scripts/check_release_artifacts.py`
- `scripts/update_ledger.py`
- `src/spacemap/__init__.py`
- `src/spacemap/doctor.py`
- `src/spacemap/marker_config.py`
- `src/spacemap/userpaths.py`
- `src/spacemap_gui/app.py`
- `src/spacemap_gui/launch.py`
- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/pipeline_folders.py`
- `src/spacemap_gui/tabs/tab6_batch_correction.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/common.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/build/THIRD_PARTY_NOTICES.txt`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/build/assets/index-90cf3627.js`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/build/assets/plotly.min-6d21266c.js`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/build/index.html`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/package-lock.json`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/package.json`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/src/index.js`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/src/index.ts`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/vendor-licenses/streamlit-component-lib-LICENSE.txt`
- `src/spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/write-notices.mjs`
- `tests/test_doctor.py`
- `tests/test_environment.py`
- `tests/test_gui_state_location.py`
- `tests/test_marker_config.py`
- `tests/test_pipeline_folders.py`
- `tests/test_repository_privacy.py`

</details>

<details><summary><code>6ab5193</code> feat(rois): a crop can keep its whole box instead of blanking outside the shape</summary>

- `src/spacemap/registration.py`
- `tests/test_registration_metadata.py`

</details>

<details><summary><code>1fee954</code> feat(images): ROI crops and background-removed images keep their parent's metadata</summary>

- `CHANGELOG.md`
- `src/spacemap/background_detection.py`
- `src/spacemap/background_subtraction_utils.py`
- `src/spacemap/image_metadata.py`
- `src/spacemap/registration.py`
- `src/spacemap/utils.py`
- `tests/test_registration_metadata.py`

</details>

<details><summary><code>aa99c1f</code> refactor(tab5): segmentation region and cell tagging run from saved settings; annotation chosen by hand</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/tabs/cell_tagging_ui.py`
- `src/spacemap_gui/tabs/segmentation_spatial_ui.py`
- `src/spacemap_gui/tabs/tab5_segmentation.py`
- `tests/test_cell_tagging_ui.py`
- `tests/test_segmentation_region.py`

</details>

<details><summary><code>095d2bf</code> feat(tab1): any CODEX image can be paired with any H&E image by hand</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/tabs/tab1_registration.py`
- `tests/page_smoke.py`
- `tests/test_tab1_pairing.py`

</details>

<details><summary><code>272039a</code> feat(registration): the merged image keeps all of the CODEX metadata</summary>

- `CHANGELOG.md`
- `src/spacemap/image_metadata.py`
- `src/spacemap/registration.py`
- `src/spacemap/utils.py`
- `src/spacemap_gui/cell_visualizer.py`
- `src/spacemap_gui/tabs/tab1_registration.py`
- `tests/test_registration_metadata.py`

</details>

<details><summary><code>7717de7</code> perf(tab8): cell-cell interactions count pairs with bincount</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/tabs/tab8_analysis/cell_cell_interaction.py`
- `tests/test_cell_cell_interaction.py`

</details>

<details><summary><code>c26d11a</code> feat(tab6): cyCombine's own batch correction, exact, as the default</summary>

- `CHANGELOG.md`
- `src/spacemap/batch_correction_utils.py`
- `src/spacemap/cycombine.py`
- `src/spacemap_gui/tabs/tab6_batch_correction.py`
- `tests/cycombine_r_reference.json`
- `tests/test_cycombine.py`

</details>

<details><summary><code>fe018b2</code> fix(assistant): "stop the model with SPACEMAP" can be saved</summary>

- `src/spacemap/ai_settings.py`
- `tests/test_model_runtime_nodes.py`

</details>

<details><summary><code>20ad9a5</code> fix(assistant): the local model server on a cluster: one record per node, no leftovers, a free port</summary>

- `src/spacemap_assistant/model_runtime.py`
- `src/spacemap_assistant/model_service/guard.py`
- `tests/test_model_runtime_nodes.py`

</details>

<details><summary><code>4de20b9</code> feat(assistant): one AI setup page; the chosen model starts when you ask</summary>

- `docs/ai-assistant.md`
- `src/spacemap_assistant/model_runtime.py`
- `src/spacemap_assistant/model_search.py`
- `src/spacemap_gui/components/ai_setup.py`
- `src/spacemap_gui/components/assistant_panel.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `tests/test_ai_install_help.py`
- `tests/test_ai_setup.py`
- `tests/test_assistant_ui_smoke.py`

</details>

<details><summary><code>0f1b0c1</code> build: one environment for SPACEMAP, the viewer and the assistant; spacemap doctor</summary>

- `docs/installation.md`
- `environment.yml`
- `pyproject.toml`
- `requirements-lock.txt`
- `src/spacemap/ai_settings.py`
- `src/spacemap/doctor.py`
- `src/spacemap_gui/launch.py`
- `tests/test_doctor.py`

</details>

<details><summary><code>79c406b</code> fix: Tab 3 reads the last step's images; [local directory] and [local directory] are one folder</summary>

- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/pipeline_folders.py`
- `src/spacemap_gui/tabs/tab3_roi_extraction.py`
- `tests/test_pipeline_folders.py`

</details>

<details><summary><code>4922cff</code> fix(assistant): the local model answers; live download progress</summary>

- `src/spacemap_assistant/model_client.py`
- `src/spacemap_assistant/model_service/server.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `tests/test_ai_install_help.py`
- `tests/test_environment.py`

</details>

<details><summary><code>7ea0ff8</code> fix: keep SPACEMAP's settings out of the node-local /tmp folder</summary>

- `docs/architecture.md`
- `src/spacemap/userpaths.py`
- `tests/test_environment.py`

</details>

<details><summary><code>330c0a9</code> fix(assistant): keep Python's own itertools.pairwise so the model server loads</summary>

- `src/spacemap/utils.py`
- `tests/test_environment.py`

</details>

<details><summary><code>2a8050b</code> fix(assistant): clear instructions for installing what the local model needs</summary>

- `src/spacemap/ai_settings.py`
- `src/spacemap_assistant/model_installer.py`
- `src/spacemap_assistant/model_runtime.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `tests/test_ai_install_help.py`

</details>

<details><summary><code>6371845</code> test: every page must still render (guard for changes elsewhere)</summary>

- `tests/page_smoke.py`
- `tests/test_pages_render.py`

</details>

<details><summary><code>afd8ad2</code> fix(tab3): refill blank marker names and black colours from the image</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/tabs/tab3_roi_extraction.py`
- `tests/test_tab3_marker_defaults.py`

</details>

<details><summary><code>aa5a947</code> fix: run log crashed on settings keyed by numbers</summary>

- `src/spacemap_gui/run_log.py`
- `tests/test_run_log.py`

</details>

<details><summary><code>c0b12ec</code> fix: switching to a moved project no longer loops forever</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/app.py`
- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/tabs/tab0_project.py`
- `tests/test_project_moves.py`

</details>

<details><summary><code>cb65845</code> fix: keep data out of the project file, and shrink files it already bloated</summary>

- `CHANGELOG.md`
- `src/spacemap_gui/config_repair.py`
- `src/spacemap_gui/persistence.py`
- `tests/test_config_size_guard.py`

</details>

<details><summary><code>dcd8d79</code> feat: one run log per tab, and phenotype saves without rewriting the CSV</summary>

- `.gitignore`
- `CHANGELOG.md`
- `docs/handover.md`
- `src/spacemap/dataio.py`
- `src/spacemap/pending_columns.py`
- `src/spacemap_gui/app.py`
- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/process_log.py`
- `src/spacemap_gui/run_log.py`
- `src/spacemap_gui/tabs/tab0_project.py`
- `src/spacemap_gui/tabs/tab1_registration.py`
- `src/spacemap_gui/tabs/tab2_background_subtraction.py`
- `src/spacemap_gui/tabs/tab3_roi_extraction.py`
- `src/spacemap_gui/tabs/tab5_segmentation.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/common.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/leiden_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/resolve_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/som_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/vortex_clustering.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_paper_figures.py`
- `src/spacemap_gui/ui_utils.py`
- `src/spacemap_workflows/leiden_job.py`
- `tests/test_pending_columns.py`
- `tests/test_run_log.py`

</details>

<details><summary><code>1cd78dd</code> fix: save every setting, unify Tab 7 layout and terminal output</summary>

- `CHANGELOG.md`
- `docs/handover.md`
- `notebooks/Step5_batch_correction.ipynb`
- `notebooks/Step6C_phenotype_threshold.ipynb`
- `src/spacemap/acc_utils.py`
- `src/spacemap/annotation_utils.py`
- `src/spacemap/background_subtraction_utils.py`
- `src/spacemap/batch_correction_utils.py`
- `src/spacemap/comparison_utils.py`
- `src/spacemap/config.py`
- `src/spacemap/model_catalog.py`
- `src/spacemap/ph_leiden_utils.py`
- `src/spacemap/ph_scimap_utils.py`
- `src/spacemap/ph_som_utils.py`
- `src/spacemap/ph_thres_utils.py`
- `src/spacemap/ph_vortex_utils.py`
- `src/spacemap/registration.py`
- `src/spacemap/segmentation_cellpose.py`
- `src/spacemap/spatial_analysis_utils.py`
- `src/spacemap/tissue_continuity.py`
- `src/spacemap/utils.py`
- `src/spacemap_assistant/blur_action.py`
- `src/spacemap_gui/app.py`
- `src/spacemap_gui/cell_visualizer.py`
- `src/spacemap_gui/components/assistant_panel.py`
- `src/spacemap_gui/components/assistant_setup.py`
- `src/spacemap_gui/file_browser.py`
- `src/spacemap_gui/persistence.py`
- `src/spacemap_gui/tabs/cell_tagging_ui.py`
- `src/spacemap_gui/tabs/legacy_cell_annotation.py`
- `src/spacemap_gui/tabs/tab0_project.py`
- `src/spacemap_gui/tabs/tab1_registration.py`
- `src/spacemap_gui/tabs/tab2_background_subtraction.py`
- `src/spacemap_gui/tabs/tab3_roi_extraction.py`
- `src/spacemap_gui/tabs/tab4_annotation/common.py`
- `src/spacemap_gui/tabs/tab5_segmentation.py`
- `src/spacemap_gui/tabs/tab6_batch_correction.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/__init__.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/cell_visualization.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/common.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/consensus_ml.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/leiden_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/mr_ml_phenotyping.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/resolve_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/scimap_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/som_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/subphenotyping.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/threshold_gating.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/visualization.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/vortex_clustering.py`
- `src/spacemap_gui/tabs/tab8_analysis/clonality.py`
- `src/spacemap_gui/tabs/tab8_analysis/common.py`
- `src/spacemap_gui/tabs/tab8_analysis/density_analysis.py`
- `src/spacemap_gui/tabs/tab8_analysis/feature_screening.py`
- `src/spacemap_gui/tabs/tab8_analysis/feature_screening_core.py`
- `src/spacemap_gui/tabs/tab8_analysis/roi_explorer.py`
- `src/spacemap_gui/tabs/tab8_analysis/triads.py`
- `src/spacemap_gui/tabs/tab9_visualization/actual_tissue.py`
- `src/spacemap_gui/tabs/tab9_visualization/common.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_paper_figures.py`
- `src/spacemap_gui/tabs/tab9_visualization/f1_strict_validation_ui.py`
- `src/spacemap_gui/tabs/tab9_visualization/proportion.py`
- `src/spacemap_gui/terminal.py`
- `src/spacemap_gui/ui_utils.py`
- `tests/test_resolve_z1_mode.py`
- `tests/test_widget_keys_persist.py`

</details>

<details><summary><code>324de8b</code> fix: allowlist the docs that describe the path rule</summary>

- `scripts/check_no_local_paths.py`

</details>

<details><summary><code>252ad0e</code> docs: add a handover document and a publishing procedure</summary>

- `README.md`
- `docs/handover.md`
- `docs/publishing-changes.md`
- `docs/working-with-this-repository.md`

</details>

<details><summary><code>d7cc4eb</code> feat: rank a cell's top markers by percentile, not normalized score</summary>

- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`

</details>

<details><summary><code>11ed7ce</code> fix: recognise button keys that begin and end with a placeholder</summary>

- `src/spacemap_gui/persistence.py`

</details>

<details><summary><code>2426062</code> fix: move the label commit control out of the form, and stop project data being committed</summary>

- `.github/workflows/ci.yml`
- `.gitignore`
- `docs/LEDGER.md`
- `scripts/check_no_local_paths.py`
- `scripts/check_streamlit_forms.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`

</details>

<details><summary><code>f574f68</code> docs: add a guide for using this version on real data</summary>

- `README.md`
- `docs/working-with-this-repository.md`

</details>

<details><summary><code>01ec57f</code> fix: name the two save actions for what they actually do</summary>

- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`

</details>

<details><summary><code>cc428ef</code> fix: put the commit control where labels are saved, and say where they went</summary>

- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`

</details>

<details><summary><code>0110154</code> perf: cache the segmentation index, and store labels without rewriting the table</summary>

- `.gitignore`
- `src/spacemap/dataio.py`
- `src/spacemap/labels.py`
- `src/spacemap_gui/cell_visualizer.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`
- `src/spacemap_gui/ui_utils.py`
- `tests/test_label_cell_identity.py`
- `tests/test_manual_reference_clear_label.py`

</details>

<details><summary><code>64a4808</code> fix: correct loader dtypes, and normalize only the cells shown</summary>

- `src/spacemap/dataio.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`
- `tests/test_dataio_dtypes.py`

</details>

<details><summary><code>dd094d7</code> perf: read only the columns a screen needs, from a columnar copy</summary>

- `.gitignore`
- `src/spacemap/dataio.py`
- `src/spacemap/ph_accel_utils.py`
- `src/spacemap/segmentation_cellpose.py`
- `src/spacemap_gui/cell_visualizer.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/__init__.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/leiden_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/mr_ml_phenotyping.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/resolve_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/scimap_clustering.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/tools.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/visualization.py`
- `src/spacemap_gui/tabs/tab7_phenotyping/vortex_clustering.py`
- `src/spacemap_gui/tabs/tab8_analysis/common.py`
- `src/spacemap_gui/tabs/tab9_visualization/__init__.py`
- `src/spacemap_gui/tabs/tab9_visualization/common.py`
- `src/spacemap_gui/tabs/tab9_visualization/pairwise.py`
- `src/spacemap_gui/tabs/tab9_visualization/proportion.py`
- `src/spacemap_gui/tabs/tab9_visualization/umap_explorer.py`
- `src/spacemap_gui/tabs/tab9_visualization/visual_validation.py`
- `src/spacemap_gui/ui_utils.py`

</details>

<details><summary><code>7335961</code> docs: record the submitted manuscript's author list</summary>

- `CITATION.cff`

</details>

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

