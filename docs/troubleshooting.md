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
