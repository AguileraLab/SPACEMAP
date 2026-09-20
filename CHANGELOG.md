# Changelog

Notable changes to SPACEMAP, newest first. This file is written by hand and
describes what each release means for users.

For the complete record of every commit — who made it and why — see
[docs/LEDGER.md](docs/LEDGER.md), which is generated from the Git history.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
