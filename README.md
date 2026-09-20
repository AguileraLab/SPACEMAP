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
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Quick start

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP

conda env create -f environment.yml
conda activate spacemap
pip install -e .

spacemap
```

Open <http://localhost:8501> and start at **Tab 0 — Project Workspace**.

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
| **6** | Batch correction | Harmonize marker intensities across batches (ComBat) |
| **7** | Phenotyping | Leiden, SOM, thresholds, SCIMAP, VORTEX, ML consensus |
| **8** | Spatial analysis | Neighbourhoods, distances, interactions, tissue continuity |
| **9** | Visualization | Spatial maps, heatmaps and publication figures |

Every stage writes into the project's `outputs/` tree, so stages compose and
you can stop and resume anywhere.

Prefer scripting? The scientific core has no GUI dependency:

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

A fresh install is a pipeline-only application. No AI packages are imported, no
model runs, nothing contacts the network, and there is no AI page.

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

With a local model, **nothing leaves the machine**: the server binds to
loopback, requires a locally generated key, and runs with proxy variables
stripped. Every action the assistant proposes needs your approval before it
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
| [Installation](docs/installation.md) | Conda, pip, Docker, offline, where files go |
| [AI assistant](docs/ai-assistant.md) | Enabling it, installing models, no-GPU setups, privacy |
| [Architecture](docs/architecture.md) | How the packages fit together and why the assistant is removable |
| [Troubleshooting](docs/troubleshooting.md) | Common failures and their fixes |
| [Contributing](CONTRIBUTING.md) | The Git workflow, in plain language |
| [Change ledger](docs/LEDGER.md) | Every change ever made, who made it, and why |
| [Changelog](CHANGELOG.md) | What each release means for users |

---

## Requirements

- **Python** 3.10 or newer
- **OS** Linux or macOS (Windows via Docker or WSL2)
- **RAM** 16 GB minimum; 64 GB+ for whole-slide work
- **GPU** optional — accelerates Cellpose segmentation and the AI assistant
- **Disk** whole-slide imaging is large; plan for your cohort, plus ~3–25 GB
  per AI model if you install one

---

## Development

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
bash scripts/install_hooks.sh      # commit template and the Why: check
pip install -e . pytest ruff

python -m pytest                   # self-contained: no data, models or GPU needed
python -m ruff check src
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

[MIT](LICENSE) © Aguilera Lab, UT Southwestern Medical Center.

SPACEMAP builds on Cellpose, DeepCell, SCIMAP, scanpy, palom and others; each
keeps its own license. Models installed through the assistant are covered by
their own terms, which you accept with the model provider, not with SPACEMAP.
