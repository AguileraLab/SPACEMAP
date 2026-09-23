# Changelog

Notable changes to SPACEMAP, newest first. This file is written by hand and
describes what each release means for users.

For the complete record of every commit — who made it and why — see
[docs/LEDGER.md](docs/LEDGER.md), which is generated from the Git history.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Interface review for publication: every setting saved with the project,
one layout for all Phenotyping methods, one terminal format, faster reads in
Tabs 7 and 9, and a Z1 option for Resolve. Grouped by tab.

### All tabs

- **Settings are saved with the project.** A setting is saved only when its
  control has a key that belongs to a tab. The controls below had none, so
  their values were lost on reload; they now save and restore. A new test
  (`tests/test_widget_keys_persist.py`) scans every tab and fails if a setting
  control could not be saved.
- **Log boxes start on.** "Create Processing Log" and "Record process events"
  are switched on whenever a project is loaded, so every run is recorded. A
  box can still be switched off for the current session.
- **One terminal format.** Every tab now prints
  `2026-09-22 14:03:11 | Tab 7 Phenotyping | Resolve | message`, with
  `Started:` / `Finished: ... (3.0 s)` lines around timed steps and `ERROR:` /
  `WARNING:` prefixes (`spacemap_gui/terminal.py`). About 200 `print` calls and
  twenty separate log helpers in ten formats now go through it. The existing
  switches still apply: `SPACEMAP_CONSOLE_LOGS`, `SPACEMAP_TAB_LOGS`,
  `SPACEMAP_PERF_LOGS`, `SPACEMAP_APP_LOGS`, `SPACEMAP_TRACE`.
- **No emoji.** About 190 emoji removed from labels, buttons, messages and
  expander titles. Icons, where they help, use Material icons
  (`:material/warning:`).
- **Dead code removed.** 154 unused imports and 14 repeated local imports.
  `spacemap/segmentation_cellpose.py` no longer imports `palom`, so Cellpose
  segmentation no longer requires it.

### Tab 1 — Registration

- CODEX/H&E filename prefix and suffix used for pairing are now saved.

### Tab 2 — Background Subtraction

- Blank-subtraction input source and selected image are now saved.

### Tab 3 — ROI Extraction

- **Fixed:** TMA core-detection settings (core size, sensitivity, row gap, row
  label format, name prefix) were never saved: their keys belonged to no tab.
- **Fixed:** numbered TMA rows produced duplicate core names — row 1 column 12
  and row 11 column 2 were both `R112`. Numbered rows are now `R1_C12` and
  `R11_C2`. Lettered rows (`A1`, `B12`) are unchanged up to 26 rows.
- Plain-language help for every TMA setting.

### Tab 5 — Segmentation

- Segmentation QC plot settings (colours, bins, scale, plot type, variables)
  are now saved.

### Tab 6 — Batch Correction

- **Removed dead code:** three QC functions in
  `spacemap/batch_correction_utils.py` were defined twice, identically; the
  unused first copies are gone.

### Tab 7 — Phenotyping

- **One layout for every method.** Leiden, SOM, Vortex, Resolve, SCIMAP, MR ML
  and Consensus ML now open the same way: title, one-line summary, a collapsed
  "How this method works" guide that names the matching notebook, then
  numbered sections — 1. Input File, 2. Marker Selection, 3. Configuration,
  the method's own steps, Run — followed by "Results & Phenotype Assignment"
  and "Visual Validation of Representative Cells". Sub-Phenotyping and Tools
  use the same opening.
- **New: Resolve "Z1 score (no threshold files)".** For projects without
  threshold tables. Each raw marker is ArcSinh-transformed and turned into a
  Z1 score within each batch; a marker is positive when its Z1 score is above
  the cutoff (default 1.5, adjustable). Z2, CDF and NLog are switched off in
  this mode because they would change what the cutoff means. The same rule is
  available to notebooks as `spacemap.ph_thres_utils.apply_z1_positivity`.
- Resolve's Threshold Method help no longer describes five methods that were
  removed (Li, Yen, Mean, IsoData, Minimum).
- **Settings now saved:** the "Import Columns from External CSV" tool (anchors,
  shift tolerance, columns to transfer, renames).
- **Faster:** batch-ID lists, phenotype-name scans and column-import anchors
  read single columns through the Parquet fast copy instead of parsing the
  whole CSV (minutes to about a second on a cohort-sized table).
- **Removed dead code:** two SCIMAP helper functions in
  `spacemap/ph_scimap_utils.py` were defined twice with different signatures;
  only the second copy was ever used, and the first is gone.
- Plain-language help for Leiden's GPU option and the UMAP subsample size.
- Leiden, SOM and Vortex column pickers no longer log Streamlit's "value set
  twice" warning on every render.

### Tab 8 — Spatial Analysis

- **Removed a control that did nothing:** Density Panel 2's "Zone area buffer
  percentile" slider was never read — Panel 2 uses Panel 1's tissue area.

### Tab 9 — Visualization

- **Fixed:** Paper Figures cached the unified table by path only, so an edited
  or re-exported table kept showing the old rows until restart. The cache now
  follows the file, and reads use the Parquet fast copy.

### Notebooks

- **Fixed `Step5_batch_correction.ipynb`:** cell 6 contained a block spliced
  from the loading loop, so the cell could not run, and the advertised
  `'column'` batch-ID strategy always failed. The block is back in the loading
  loop and the summary prints the batch column.
- **Fixed `Step6C_phenotype_threshold.ipynb`:** 59 source lines had lost their
  line breaks, so `PHENOTYPE_EXCLUSIONS` and the exclusion step were silently
  merged into comments and never ran. Exclusion markers apply again, matching
  the Resolve tab's "Must not have" column. Added `USE_Z1_POSITIVITY` /
  `Z1_CUTOFF`, matching the new Resolve option.

## [0.3.0] — 2026-09-19

First public release.

### Added

- `spacemap` command-line entry point that launches the interface and applies
  the numeric thread caps the pipeline needs.
- A user-writable state root (`spacemap.userpaths`): settings, caches, logs and
  models live under `~/.spacemap` (or `$SPACEMAP_HOME`) instead of inside the
  installed package. Existing in-package state is migrated on first run.
- A model catalog (`spacemap.model_catalog`) that makes model choice data
  rather than code, with a built-in list and a user-editable override.
- In-GUI model installation: browse a catalog, download from Hugging Face with
  a progress bar, import weights from a folder on an offline machine, add a
  model the catalog does not know about, and remove models to reclaim space.
- Hardware detection (`spacemap.hardware`) that reports what the machine can
  run and chooses a load precision.
- Optional dependency extras: `ai`, `gguf`, `deepcell`, `notebook`, `all`.
- GGUF backend, so a quantized model can be served through llama.cpp on a
  machine with no GPU.
- A change ledger (`docs/LEDGER.md`) generated from the Git history, a
  `commit-msg` hook requiring every commit to record why it was made, and a CI
  check that the ledger is current.
- CI covering lint, tests on Python 3.10–3.12, a no-extras install that proves
  the pipeline works without the AI packages, a guard against absolute paths
  from a contributor's machine, and a check that notebook outputs are stripped.
- Contributor documentation, issue and pull-request templates, MIT license and
  citation metadata.

### Changed

- **The AI assistant is optional and off by default.** Its page is not
  registered and its modules are not imported until it is switched on in
  Tab 0. `SPACEMAP_ASSISTANT=off` disables it for an entire installation.
- The local model server accepts any model directory instead of one fixed
  model, on a configurable port, and handles both multimodal and text-only
  models.
- `deepcell` moved out of the base dependencies into an extra, so a normal
  install no longer pulls TensorFlow.
- The Slurm integration is site-neutral: partitions are ranked by what Slurm
  reports, with optional site preferences, and the feature hides itself when
  Slurm is not installed.
- Notebooks build every path from a `DATA_DIR` configuration cell, and their
  saved outputs are stripped.
- The test suite is self-contained: it creates its own temporary state root and
  fixtures, and needs no research data, no models and no GPU.

### Fixed

- **The local model server no longer refuses to start without a GPU.** It
  loads on CPU in float32 instead. Quantized precisions still require CUDA and
  now say so plainly rather than failing.
- The model server ships inside the package, so it is present in a `pip`
  install and not only in a git checkout.
- The assistant's path guard no longer assumes a fixed directory depth, so it
  is correct for an installed package as well as an editable checkout.

### Removed

- Institution-specific paths, cluster names, account identifiers and internal
  planning documents.
- Committed interface state containing one developer's absolute paths and
  project names.

[Unreleased]: https://github.com/AguileraLab/SPACEMAP/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AguileraLab/SPACEMAP/releases/tag/v0.3.0
