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

The default image uses CPU-only PyTorch wheels. For NVIDIA GPU use:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   on the Linux Docker host and enable the `deploy` block in `docker-compose.yml`.
2. Select CUDA-enabled PyTorch wheels and rebuild. An empty index uses pip's
   default Linux wheels; for a particular CUDA build, set the index URL from
   the [PyTorch installer](https://pytorch.org/get-started/locally/) instead.
   Check that the selected build supports your GPU and host driver.

   ```bash
   export SPACEMAP_TORCH_INDEX_URL=""
   docker compose build
   docker compose run --rm spacemap python /opt/spacemap/scripts/check_runtime.py --device cuda
   docker compose up
   ```

The GPU check must pass; merely starting the GUI does not establish GPU access.
Keep the index setting when rebuilding so the image does not revert to CPU
wheels. See [Docker's GPU instructions](https://docs.docker.com/compose/how-tos/gpu-support/).
Compose publishes port 8501; use this setup on your workstation or a trusted network.

### Validate an installation

From the repository root, with the installed environment active:

```bash
python -m pip check
python scripts/check_installed_app.py
python scripts/check_runtime.py --device cpu
```

On an allocated NVIDIA GPU, run:

```bash
python scripts/check_runtime.py --device cuda --report ../gpu-validation.json
python -m pytest -q tests/test_spatial_distance_acceleration.py
```

The runtime check fails when CUDA is unavailable. It exercises compiled
PyTorch/torchvision operators and compares SPACEMAP's GPU distances with GEOS;
therefore it cannot silently report a CPU fallback as a GPU pass. Optional
`--cellpose-weights /path/to/cellpose3-model` also runs SPACEMAP's Cellpose wrapper
on a small synthetic image using an existing model file. This check downloads
no model or study data. Its JSON report records package versions and the model
checksum without recording local paths. Synthetic inference checks execution,
not biological segmentation accuracy or large-image memory requirements.

To test the actual container, install Docker and Python 3.10 or newer on the
host, then run from the clone:

```bash
docker compose config --quiet
docker build --tag spacemap:validation .
python scripts/check_container.py --image spacemap:validation
```

The container check verifies dependencies, CPU operators, opening the installed
GUI and creating a project, non-root execution, HTTP health and writing to
persistent volumes. It creates and removes its own temporary container and
volumes. It does not touch existing SPACEMAP projects or application volumes.
The same check is included in GitHub Actions as **Docker CPU build and startup**.
Open a pull request to run CI on a review branch; ordinary branch pushes only
trigger the current workflow when the branch is `main`. After the workflow is
on `main`, **Actions → CI → Run workflow** can also start it manually.

### Other platforms

Linux x86-64 is the validated native platform. The following synthetic checks
passed on 1 October 2026, with Python 3.10.21 and Cellpose 3.1.1.3:

| Runtime | PyTorch / torchvision | Result |
| --- | --- | --- |
| CPU | 2.14.1+cpu / 0.29.1+cpu | Compiled operators, distances and Cellpose inference passed |
| NVIDIA L4 | 2.14.0+cu130 / 0.29.0+cu130 | CUDA operators, Cellpose inference and all 6 distance tests passed |
| NVIDIA H200 NVL | 2.14.0+cu130 / 0.29.0+cu130 | CUDA operators, Cellpose inference and all 6 distance tests passed |
| NVIDIA RTX PRO 6000 Blackwell | 2.14.0+cu130 / 0.29.0+cu130 | CUDA operators, Cellpose inference and all 6 distance tests passed |

These GPU runs used an existing native environment, not the Docker image.
Docker execution and macOS/Windows installations still require validation.
A Linux container test alone does not validate macOS, Windows, ARM hardware or
a different GPU model.

- **Windows:** use WSL2 with the Linux instructions, or Docker Desktop's WSL2
  backend. Run the checks above inside that setup, then open the GUI in the
  Windows browser and create a project in the intended mounted folder. NVIDIA
  passthrough has additional [Docker Desktop prerequisites](https://docs.docker.com/desktop/features/gpu/).
- **macOS:** native installation remains unvalidated. On Apple Silicon, try
  the x86-64 Docker image with `export DOCKER_DEFAULT_PLATFORM=linux/amd64`
  before building. This uses emulation and still needs a real Mac test. The
  CUDA check does not validate Apple GPU/MPS support.
- **Each host:** verify the chosen project folder is writable, save and reopen
  a project, restart the application, and confirm that its files and settings
  remain. A passing named-volume check does not guarantee permissions for
  every host bind mount.

Record the OS, architecture, package versions, GPU model (when used) and results
before adding a platform to the supported list. AI model inference, DeepCell
and GGUF backends require their own checks; these runtime tests do not validate
those optional configurations.

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
