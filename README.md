# SPACEMAP

**Spatial multiplex imaging analysis, end to end.**

SPACEMAP takes multiplexed tissue images from raw acquisition to spatial
statistics: registration, background subtraction, ROI extraction, tissue
annotation, cell segmentation, batch correction, immune phenotyping, spatial
analysis and publication figures — through a graphical interface, a set of
reusable Python modules, or notebooks.

An optional, local-first AI assistant can help you drive the pipeline. It is
**switched off by default** and everything works without it.

[![CI](https://github.com/AguileraLab/SPACEMAP/actions/workflows/ci.yml/badge.svg)](https://github.com/AguileraLab/SPACEMAP/actions/workflows/ci.yml)
[![License: Academic research only](https://img.shields.io/badge/License-Academic%20research%20only-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

---

## Quick start

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP

conda env create -f environment.yml
conda activate spacemap
spacemap doctor

spacemap
```

Open <http://localhost:8501> and start at **Tab 0 — Project Workspace**.
Follow **[Your first project](docs/first-project.md)** to create a project,
connect your images, and move through the pipeline. The clone contains the
software and notebooks; study images and model weights are separate downloads.
While this repository is private, cloning requires collaborator access.

The setup uses **Python 3.10** and includes JupyterLab. AI packages and DeepCell
are optional. `spacemap doctor` reports `PROBLEM` for issues to fix; a `NOTE`
about no GPU or missing optional AI packages is expected on a CPU-only setup.

Prefer Docker, or installing with pip into an environment you already manage?
See **[docs/installation.md](docs/installation.md)**.

---

## The pipeline

| Tab | Stage | What it does |
| --- | --- | --- |
| **0** | Project workspace | Create or load a project, set directories, configure the optional AI assistant |
| **1** | Registration | Align CODEX rounds to each other and to H&E |
| **2** | Background subtraction | Detect and remove autofluorescence and background |
| **3** | ROI extraction | Cut regions of interest from whole-slide images |
| **4** | Annotation | Annotate tissue compartments and zones |
| **5** | Segmentation | Segment cells with Cellpose, or DeepCell/Mesmer |
| **6** | Batch correction | Harmonize marker intensities across batches (ComBat and cyCombine) |
| **7** | Phenotyping | Leiden, SOM, thresholds, SCIMAP, VORTEX, ML consensus |
| **8** | Spatial analysis | Neighbourhoods, distances, interactions, tissue continuity |
| **9** | Visualization | Spatial maps, heatmaps and publication figures |

Every stage writes into the project's `outputs/` tree, so stages compose and
you can stop and resume anywhere.

Prefer scripting? Call the scientific modules without starting the GUI:

```python
from spacemap.config import get_config
from spacemap.segmentation_cellpose import SegmentationConfig, segment_image_cellpose

config = get_config("/path/to/project", create_directories=True)
segment_image_cellpose("image.ome.tiff", config.segmentation_dir, SegmentationConfig())
```

The `notebooks/` folder carries the same workflow step by step, ready to edit.
Each one opens with a configuration cell; point `DATA_DIR` at your data and the
rest follows.

---

## The AI assistant is optional

A fresh install starts with the assistant disabled and does not load a language
model. Installing packages and downloading Cellpose or AI model weights require
network access. Download the needed weights before using the pipeline offline.

**To turn it on:** Tab 0 → *AI Assistant (optional)* → **Enable AI assistant**.

From that panel you can, without editing a file or leaving the browser:

- see what your machine can run — the panel reads your CPU, memory and GPU and
  says so plainly;
- install the AI packages (it prints the exact command for *your* interpreter);
- browse a **model library**, download a model with a progress bar, and pick
  which one is active;
- install weights from a folder instead, on a machine with no network;
- **add any model the catalog does not know about** — a Hugging Face repository
  or a GGUF file — so new models need no code change;
- move the model store to a bigger disk;
- turn the whole thing off again.

**It runs without a GPU.** A machine with no accelerator loads the model on the
CPU. For that case the catalog includes a ~2.5 GB quantized model that is
practical on a laptop.

| Model | Size | Needs | Reads images |
| --- | --- | --- | --- |
| Qwen3 4B Instruct (Q4_K_M) | ~2.5 GB | No GPU | No |
| Gemma 4 12B Instruct | ~24 GB | ~28 GB GPU | Yes |
| Llama 3.1 8B Instruct | ~16 GB | ~20 GB GPU | No |

Local model inference runs on your machine: the model server binds to
loopback, requires a locally generated key, and runs with proxy variables
stripped. Downloads and any explicitly enabled online tools still use the network. Every action the assistant proposes needs your approval before it
runs, and it can never write to the installed source or your configuration.

To disable it across a shared installation regardless of the setting:

```bash
export SPACEMAP_ASSISTANT=off
```

Full details: **[docs/ai-assistant.md](docs/ai-assistant.md)**.

---

## Where SPACEMAP keeps its files

Nothing is written inside the installed package. Everything lives under
`~/.spacemap` (settings, caches, logs, models), and you can move it:

```bash
export SPACEMAP_HOME=/data/shared/spacemap
export SPACEMAP_MODELS_DIR=/mnt/bigdisk/spacemap-models   # or just the models
```

Your imaging data stays wherever you keep it; SPACEMAP only records the paths.

---

## Documentation

| Page | |
| --- | --- |
| [Your first project](docs/first-project.md) | Create a project, prepare inputs, and follow the stages |
| [Installation](docs/installation.md) | Conda, pip, Docker, offline, where files go |
| [AI assistant](docs/ai-assistant.md) | Enabling it, installing models, no-GPU setups, privacy |
| [Architecture](docs/architecture.md) | How the packages fit together and why the assistant is removable |
| [Troubleshooting](docs/troubleshooting.md) | Common failures and their fixes |
| [Handover](docs/handover.md) | What SPACEMAP is, what has been done to it, and the lessons behind it |
| [Publishing changes](docs/publishing-changes.md) | The exact procedure for getting an edit onto GitHub safely |
| [Working with this repository](docs/working-with-this-repository.md) | Using it on your own data, keeping data out of Git, sending changes back |
| [Contributing](CONTRIBUTING.md) | The Git workflow, in plain language |
| [Change ledger](docs/LEDGER.md) | Every change ever made, who made it, and why |
| [Changelog](CHANGELOG.md) | What each release means for users |

---

## Requirements

- **Python** 3.10 (the supported version for the current imaging dependencies)
- **OS** Linux is the tested platform; macOS and Windows via Docker/WSL2 need platform validation
- **RAM** 16 GB minimum; 64 GB+ for whole-slide work
- **GPU** optional — accelerates Cellpose segmentation and the AI assistant
- **Disk** whole-slide imaging is large; plan for your cohort, plus ~3–25 GB
  per AI model if you install one

---

## Development

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
conda env create -f environment.yml
conda activate spacemap
python -m pip install --upgrade pip
bash scripts/install_hooks.sh      # commit template and the Why: check
python -m pip install -e . --group dev

python -m pytest                   # self-contained: no data, models or GPU needed
python -m ruff check src scripts
```

Every change is recorded in [docs/LEDGER.md](docs/LEDGER.md) — what changed,
who made it, and why — generated from the Git history and checked by CI. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## Citing SPACEMAP

If SPACEMAP contributes to work you publish, please cite it. Machine-readable
metadata is in [CITATION.cff](CITATION.cff); GitHub renders a formatted
citation from the sidebar.

---

## License

SPACEMAP is available for academic research use only under the
[UT Southwestern license](LICENSE). Commercial use and redistribution,
including any use or redistribution by a for-profit entity, are prohibited.

Copyright (c) 2025-2026 The University of Texas Southwestern Medical Center.
All rights reserved.

SPACEMAP builds on Cellpose, DeepCell, SCIMAP, scanpy, palom and others; each
keeps its own license. Models installed through the assistant are covered by
their own terms, which you accept with the model provider, not with SPACEMAP.
