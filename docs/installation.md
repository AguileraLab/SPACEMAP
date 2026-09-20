# Installing SPACEMAP

SPACEMAP installs in three ways. Pick one:

| Route | Good for | Needs |
| --- | --- | --- |
| [Conda](#conda-recommended) | Most users. Handles the geospatial and imaging C libraries for you. | Miniconda or Mambaforge |
| [pip](#pip) | An environment you already manage. | Python 3.10+ and system libraries |
| [Docker](#docker) | A reproducible run, or a machine you cannot install onto. | Docker |

The AI assistant is **optional and off by default** in every route. Install it
later, from inside the interface — see [ai-assistant.md](ai-assistant.md).

---

## Conda (recommended)

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP

conda env create -f environment.yml
conda activate spacemap

pip install -e .
```

Start it:

```bash
spacemap
```

Then open <http://localhost:8501>.

---

## pip

SPACEMAP depends on packages that wrap C libraries (GEOS for Shapely, PROJ and
GDAL for rasterio and geopandas, OpenGL for OpenCV). Install those first.

**Debian / Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential libgl1 libglib2.0-0 libgeos-dev \
    libproj-dev proj-bin gdal-bin libgdal-dev libhdf5-dev libopenblas-dev
```

**Red Hat / Rocky / Fedora**

```bash
sudo dnf install -y gcc gcc-c++ mesa-libGL glib2 geos-devel proj-devel gdal-devel hdf5-devel openblas-devel
```

**macOS** (Homebrew)

```bash
brew install geos proj gdal hdf5 openblas
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
spacemap
```

---

## Docker

No Python setup at all. Your data is mounted into the container.

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP

# Point this at the folder holding your projects.
export SPACEMAP_PROJECTS_DIR=/path/to/your/projects

docker compose up --build
```

Open <http://localhost:8501>. Everything the application writes — settings,
logs, downloaded models — persists in the `spacemap-home` Docker volume, so it
survives a rebuild.

To include the AI assistant's dependencies in the image:

```bash
docker compose build --build-arg SPACEMAP_EXTRAS="ai"
```

To use a GPU, uncomment the `deploy:` block in `docker-compose.yml` (requires
the NVIDIA Container Toolkit).

---

## Optional extras

The base install runs the full scientific pipeline and the GUI. These are
separate because they are large, slow to install, or platform-specific:

```bash
pip install -e ".[ai]"        # run a local AI model (transformers, fastapi)
pip install -e ".[gguf]"      # a small quantized model via llama.cpp (no GPU needed)
pip install -e ".[deepcell]"  # DeepCell/Mesmer segmentation (pulls TensorFlow)
pip install -e ".[notebook]"  # JupyterLab for the notebooks
pip install -e ".[all]"       # everything
```

**You do not need to install the AI extras by hand.** Tab 0 shows the exact
command for your environment and re-checks when you have run it.

DeepCell is optional on purpose: it pulls TensorFlow, which is a large and
version-sensitive install. Cellpose segmentation is included in the base
install and needs nothing extra.

---

## Where SPACEMAP keeps its files

Nothing is written inside the installed package. Everything lives under one
root, by default `~/.spacemap`:

```
~/.spacemap/
├── config/           settings, model catalog, API tokens
├── state/            interface state between sessions
├── cache/            Numba, Cellpose and Hugging Face caches
├── logs/             application and model-server logs
└── models/           downloaded model weights
```

Move all of it with one variable:

```bash
export SPACEMAP_HOME=/data/shared/spacemap
```

Or redirect one directory at a time — useful when models belong on a large
disk:

```bash
export SPACEMAP_MODELS_DIR=/mnt/bigdisk/spacemap-models
```

Check where your installation is actually writing:

```python
from spacemap.userpaths import describe_layout
print(describe_layout())
```

| Variable | What it moves |
| --- | --- |
| `SPACEMAP_HOME` | Everything below, at once |
| `SPACEMAP_CONFIG_DIR` | Settings and tokens |
| `SPACEMAP_STATE_DIR` | Interface state |
| `SPACEMAP_CACHE_DIR` | Regenerable caches |
| `SPACEMAP_LOGS_DIR` | Logs |
| `SPACEMAP_MODELS_DIR` | Model weights |
| `SPACEMAP_PROJECTS_DIR` | Default location for new projects |
| `SPACEMAP_ASSISTANT` | Set to `off` to disable the AI assistant entirely |
| `SPACEMAP_DATA_DIR` | Default data root used by the notebooks |

---

## Verifying the install

```bash
spacemap --version
python -m pytest            # from a clone
```

The test suite is self-contained: it creates its own temporary
`SPACEMAP_HOME` and needs no data, no models and no GPU.

---

## Notebooks

```bash
pip install -e ".[notebook]"
export SPACEMAP_DATA_DIR=/path/to/your/data
jupyter lab notebooks/
```

Every notebook opens with a configuration cell that builds all its paths from
`DATA_DIR`. Edit that one cell, or set `SPACEMAP_DATA_DIR`, and the rest of the
notebook follows.

Before committing a notebook, strip its outputs:

```bash
pip install nbstripout
nbstripout --install     # once per clone
```

---

## Uninstalling

```bash
pip uninstall spacemap-ai
rm -rf ~/.spacemap        # settings, caches and downloaded models
```

Your project data is never stored inside `~/.spacemap` unless you put it there.
