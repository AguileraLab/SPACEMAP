# Installing SPACEMAP

Use **Python 3.10**. Registration requires scikit-image 0.19.x, whose prebuilt
wheels stop at Python 3.10. Package metadata, Conda, Docker and CI use the same
supported version. Linux is the tested platform; macOS and Windows through
WSL2/Docker still need platform-specific validation.

The default installation includes Cellpose. AI and DeepCell/Mesmer are optional.
Study images and model weights are separate from the repository. While the
repository is private, cloning requires GitHub collaborator access.

## Conda (recommended)

Install Git and Conda/Miniforge, then run:

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
conda env create -f environment.yml
conda activate spacemap
spacemap --version
spacemap doctor
spacemap
```

Open <http://localhost:8501> and follow [Your first project](first-project.md).
Keep the terminal open; press Ctrl+C to stop the application.

Conda supplies Python and the OpenSlide 3.x native library. Pip installs SPACEMAP
and JupyterLab in editable mode, so no second installation command is needed.
AI packages and developer tools are not part of this recipe. On Linux, OpenCV
also needs `libGL` and GLib; see the native-library commands below if an import
reports a missing shared library.

`spacemap doctor` reports `OK`, `NOTE` or `PROBLEM`. Fix `PROBLEM` entries before
analysis. A missing GPU or optional AI packages is an expected `NOTE` on a CPU
setup. Also run `python -m pip check` to check dependency consistency.

Default Linux PyTorch packages can download several GB of CUDA libraries even
without a GPU. For a smaller CPU installation, use the pip route below.

## pip

Install native libraries first. OpenSlide supports slide formats used by
palom; OpenGL/GLib support OpenCV; GEOS, PROJ and GDAL support spatial data.

Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgl1 libglib2.0-0 libopenslide0 \
    libgeos-dev libproj-dev proj-bin gdal-bin libgdal-dev libhdf5-dev libopenblas-dev
```

Red Hat/Rocky/Fedora (packages depend on enabled system repositories):

```bash
sudo dnf install -y gcc gcc-c++ mesa-libGL glib2 openslide geos-devel \
    proj-devel gdal-devel hdf5-devel openblas-devel
```

On macOS, use the Conda route to obtain OpenSlide 3.x. The current OpenSlide
4.x library uses a different ABI and does not work with the older Python
binding required by this registration stack. The same limitation applies if
a Linux distribution supplies only OpenSlide 4.x; prefer the Conda recipe.

Create a separate environment, with Python 3.10 already installed:

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Optional on Linux: avoid CUDA downloads when using only the CPU.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e .
spacemap doctor
spacemap
```

For GPU use, omit the CPU command and select a PyTorch build compatible with
your driver using the [official installer](https://pytorch.org/get-started/locally/).
`spacemap doctor` reports whether the environment can see the GPU. Windows
users should follow Linux instructions inside WSL2, or use Docker.

## Docker

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
mkdir -p /path/to/your/projects
export SPACEMAP_PROJECTS_DIR=/path/to/your/projects
docker compose up --build
```

Open <http://localhost:8501>. In Tab 0 choose **`/workspace/projects`** as the
parent folder: that is the container's view of your host folder. The container
user needs write access to it. Settings, logs and models persist in the
`spacemap-home` volume. Stop with `docker compose down`; adding `--volumes`
would delete that saved application state.

To include AI packages:

```bash
docker compose build --build-arg SPACEMAP_EXTRAS="ai"
docker compose up
```

For GPU access, install NVIDIA Container Toolkit and enable the `deploy` block
in `docker-compose.yml`. Check GPU availability inside the container. Compose
publishes port 8501; use this setup on your workstation or a trusted network.

## Remote server or cluster

Choose persistent locations with enough disk space:

```bash
export PIP_CACHE_DIR=/path/to/work/pip-cache
export SPACEMAP_HOME=/path/to/work/spacemap-state
export SPACEMAP_PROJECTS_DIR=/path/to/work/experiments
spacemap --server.address=127.0.0.1 --server.headless=true --port 8501
```

On your laptop, in a separate terminal:

```bash
ssh -N -L 8501:127.0.0.1:8501 your_username@your_server
```

Open <http://localhost:8501> on your laptop. If the app runs on a scheduled
compute node, follow your institution's forwarding instructions to reach that
node; forwarding to a different login node is insufficient. Paths selected in
the browser refer to the server's filesystem.

## Optional tools and offline use

Run from the clone with the SPACEMAP environment active:

```bash
python -m pip install -e ".[notebook]"  # JupyterLab; already in environment.yml
python -m pip install -e ".[ai]"        # local language-model assistant
python -m pip install -e ".[gguf]"      # llama.cpp; may need C/C++ build tools
python -m pip install -e ".[deepcell]"  # alternative segmentation; pulls TensorFlow
```

DeepCell is version-sensitive and excluded on Apple Silicon by its platform
marker. Install it only when needed and rerun `pip check`. QuPath is a separate
application for reviewing tissue annotations; see the
[QuPath documentation](https://qupath.readthedocs.io/).

Enable AI through **Tab 0 → AI Assistant (optional)**; see
[AI assistant](ai-assistant.md). Initial package and model downloads require
internet access. Before working offline, prepare the matching Python/platform
environment and cache the Cellpose, DeepCell or AI weights you intend to use.
A source clone alone is not an offline installation bundle.

## Dependency versions

palom 2024.12.1 and SCIMAP impose overlapping limits on NumPy, scikit-image,
zarr, tifffile, SciPy and dask. Let pip resolve them from `pyproject.toml`;
do not upgrade one independently. palom installs `opencv-python` alongside the
headless build used by Cellpose. Their versions must match; the doctor checks
this.

`requirements-lock.txt` records an earlier **Linux x86-64/CUDA environment with
AI and notebooks**. It is an analysis snapshot, not a portable install recipe.

## Files, updates and removal

Settings, caches, logs and models normally live under `~/.spacemap`. Project
images, CSVs and outputs live in the location chosen in Tab 0. The project index
and active session now follow the same state root. Existing default-home
installations copy their old `~/.spacemap_gui` index once; an explicit
`SPACEMAP_HOME` or `SPACEMAP_STATE_DIR` starts a separate registry. Browse to load
your project if its name is not in that new registry.

| Variable | Purpose |
| --- | --- |
| `SPACEMAP_HOME` | Application settings, state, caches, logs and models |
| `SPACEMAP_CONFIG_DIR` / `SPACEMAP_STATE_DIR` | Override settings or interface state |
| `SPACEMAP_CACHE_DIR` / `SPACEMAP_LOGS_DIR` | Override caches or logs |
| `SPACEMAP_MODELS_DIR` | Override the AI model store |
| `SPACEMAP_PROJECTS_DIR` | Default parent location for new projects |
| `SPACEMAP_ASSISTANT=off` | Force the assistant off |
| `SPACEMAP_DATA_DIR` | Default data root used by notebooks |

After updating a clone, activate its environment, run
`python -m pip install -e .`, then `spacemap doctor`. Retain the release
commit/tag and environment record used for a published analysis.

To remove the package, use `python -m pip uninstall spacemap-ai`. Removing
application state is separate: review and back up settings, models and any
projects stored there before deleting it.

## Developer checks

Use a recent pip supporting dependency groups:

```bash
python -m pip install --upgrade pip
python -m pip install -e . --group dev
python -m pytest
python -m ruff check src scripts
python scripts/check_no_local_paths.py
python scripts/check_streamlit_forms.py
```

Tests create temporary data and need no study images or model downloads.
Optional AI/GPU tests skip when their requirements are absent. CI also checks
release archives. See [Contributing](../CONTRIBUTING.md) for frontend builds
and publishing steps.
