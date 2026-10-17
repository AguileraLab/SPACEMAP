# Troubleshooting

## Installation

**`pip install` fails building `rasterio`, `geopandas` or `shapely`**

The C libraries are missing. Install the system packages listed in
[installation.md](installation.md#pip), or use the Conda route, which brings
them along.

**`ImportError: libGL.so.1`**

OpenCV needs an OpenGL library even in headless use:
`sudo apt-get install -y libgl1`.

**The install tries to pull TensorFlow**

You asked for the `deepcell` extra. The base install does not include it;
Cellpose segmentation needs nothing extra.

---

## Starting up

**`spacemap: command not found`**

The environment holding SPACEMAP is not active. Activate it, or run
`python -m spacemap_gui.launch`.

**"Program is Terminated. Because you tried to allocate too many memory regions", then a crash**

OpenBLAS ran out of its fixed thread-buffer table on a many-core machine. The
`spacemap` command caps the thread pools for you; if you launch Streamlit
directly, set them yourself:

```bash
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
```

**"The project filesystem is temporarily unavailable"**

A stale handle on a network share. Retry; if it persists the share needs
remounting.

**Port 8501 is in use**

`spacemap --port 8600`

---

## Settings and state

**My directory selections disappeared after upgrading**

State moved out of the package into `~/.spacemap/state/`. It is migrated
automatically on first run. Check where it is now:

```python
from spacemap.userpaths import describe_layout
print(describe_layout())
```

**SPACEMAP cannot write its settings**

The home directory is read-only or full. Point it somewhere writable:
`export SPACEMAP_HOME=/scratch/$USER/spacemap`.

---

## Studio annotation layers

**`IsADirectoryError` for `studio_history/<annotation>.geojson`**

Studio keeps each annotation's saved edits in a directory named after the file,
including its `.geojson` suffix. An older viewer search mistook these history
directories for annotation files. This error means it tried to read a directory
as JSON; it does not establish that the working annotation is damaged.

The corrected search lists regular GeoJSON files and skips history/cache trees.
Stale directory layers receive a readable error and are excluded when restoring
added layers. Start Studio with the updated code and reopen the image to rebuild
its layer list. Keep the `studio_history` directory: it contains saved revisions.

## Shared projects: "Permission denied" for a collaborator

A project shared by a lab group (for example `Aguilera_lab`) must stay readable and writable by the whole group.
Older versions made lock and settings files writable by their creator only, so a collaborator saw errors such as
`Permission denied: .../outputs/viewer/views/<image>.json.lock`.

SPACEMAP now:

- writes new files and folders with group read and write, and nothing for others (umask 007; set
  `SPACEMAP_KEEP_UMASK=1` to keep your own);
- opens a lock read-only when it cannot write it, which is enough for locking, so old locks no longer fail;
- gives group access back to files written through private temporary files (project settings, panels, annotations,
  the model library);
- when a project opens, repairs the files *you* own in it, in the background (group read and write; folders keep
  passing the group on to new files).

Only a file's owner can change its permissions, so each collaborator's files are repaired when they open the project
with this version, or at once with:

```bash
spacemap share-project /path/to/project
```

The project folder itself should belong to the group and pass it on (`chgrp -R <group> <project>` and
`chmod g+s <project>` by its owner, once). Check with `ls -ld <project>`: the mode shows `drwxrws---` or `drwxrwsr-x`.

## The AI assistant

**There is no AI Assistant page**

That is the default. Turn it on in **Tab 0 → AI Assistant → Enable**. If the
panel says an administrator disabled it, `SPACEMAP_ASSISTANT=off` is set.

**The panel says dependencies are missing**

Run the command it prints — it uses the interpreter that is actually running
SPACEMAP — then click **Re-check dependencies**. Installing into a different
environment is the usual cause of this not clearing.

**A download fails with 401 or 403**

The model is gated. Accept its terms on Hugging Face while signed in, create a
**read** token, and save it in the panel. Accepting the terms and creating the
token are two separate steps; both are required.

**A download fails with 404**

The repository name changed. Correct it with **Add another model** using the
same `id`, which overrides the built-in entry.

**The download bar is stuck at 0%**

Progress is measured by bytes on disk and updates every couple of seconds. A
large shard can take a while to appear. Check the models folder is growing; if
the panel reports the download stopped unexpectedly, try again.

**"No CUDA GPU is visible" — but I want to run on CPU**

Older versions refused to start without a GPU. Current versions load on CPU in
float32 automatically. If you see this message, you are on an older build, or
you explicitly asked for `int8`/`int4`, which require CUDA. Choose **auto** or
**float32**, or install a GGUF model.

**Replies take minutes**

You are on CPU. That is expected. Use the Qwen3 4B GGUF model instead of a
12 B model.

**The model server will not start, and the log is unhelpful**

Find the log with `describe_layout()["logs"]`, then run the server by hand to
see the error directly:

```bash
python -m spacemap_assistant.model_service.server \
    --model-dir ~/.spacemap/models/<model> \
    --api-key-file ~/.spacemap/assistant_data/config/local_model.key
```

**"Port 8091 is in use by a process SPACEMAP did not start"**

Something else holds the port, or a previous server was not shut down cleanly.
Change the port in the assistant settings, or stop the other process.

**The Cluster tab says Slurm was not found**

Expected on a workstation. The tab only activates when `sinfo` and `sbatch` are
on the PATH.

---

## Notebooks

**The first cell fails, or paths do not exist**

Set your data location and restart the kernel:

```bash
export SPACEMAP_DATA_DIR=/path/to/your/data
```

Or edit `DATA_DIR` in the configuration cell at the top of the notebook.

**Git shows a huge diff after running a notebook**

Saved outputs. Strip them: `pip install nbstripout && nbstripout --install`.

---

## Still stuck

Open an issue with `spacemap --version`, your OS, how you installed, whether
you have a GPU, and the full error text. Describe the shape of a problem image
or CSV rather than attaching data from a study.


## Artifact model training and prediction

**`Unable to determine overlay result geometry dimension` during training**

Earlier training preparation could leave isolated points or lines where an
artifact touched the Tissue boundary. GEOS 3.13.1 can fail when comparing that
mixed result with an empty Good tissue class. Training now keeps only polygon
area in its sampling copy and skips an empty-class overlap check. Original
annotations and saved models remain unchanged. Refresh Studio and retry Train
model; adding green examples or redrawing valid polygons is not required for
this error.

**Training appears stuck while reading a few example polygons**

Update Studio and restart its process. Earlier artifact preparation could
repeatedly decompress every marker plane when a fine-resolution marker panel
exceeded the image cache. Preparation now reads windows channel by channel and
runs inside the visible training job. Check Jobs for marker progress or Cancel.
Similar polygon areas can have different costs at different detection scales.

**`invalid value encountered in sqrt` from `artifact_quality.py`**

Update and restart Studio. Training and prediction now clamp tiny negative
rounding errors in squared-energy averages before the square root. Warnings
should not be hidden with a global warning filter.

**Does artifact learning use my GPU?**

Usable CUDA devices are detected through PyTorch, with CPU fallback. GPU texture
measurements speed preparation and prediction; saved random-forest inference can
also run on GPU without retraining the trees. Forest fitting retains sklearn.
No CuPy or cuML installation is required. Hover the completed training or
prediction status for device details. The scheduler must expose an allocated GPU
to the Studio process; merely installing CUDA does not allocate one.
# Studio port and update checks (1 October 2026)

`run_interface.sh` now tries the next available port when 8502 (or the requested
port) is occupied. Use the local/network address printed by that launch. You do not
need to kill the running application or request another node. If using SSH forwarding,
forward the newly printed port as well.

A Classification `HTTP 404` on an older process means that process has not loaded
the new route; it is separate from whether the Assistant is running. Launch the updated
interface and use its new address. Running processes now keep a matching snapshot of
their UI, preventing newly edited frontend files from calling old backend routes.
The AI model can remain running while a new interface process starts.
