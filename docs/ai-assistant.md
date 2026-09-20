# The AI assistant (optional)

SPACEMAP ships with the assistant **switched off**. A fresh install is a
pipeline-only application: no AI packages are imported, no model runs, no
network call is made, and the interface shows no AI page at all.

Everything on this page is opt-in, reversible, and done from inside the
interface. You never need to edit a file.

---

## Deciding whether you want it

The assistant can answer questions about the pipeline, read your project
layout, explain what a tab does, and propose parameters for steps such as
registration and ROI extraction. **Every action it proposes needs your
approval before anything runs.** It does not make scientific decisions:
phenotype labels, tissue rejection and segmentation quality still need a human
or a deterministic check.

If you only want the pipeline, do nothing. Everything works without it.

---

## Turning it on

1. Open **Tab 0 — Project Workspace**.
2. Scroll to **AI Assistant (optional)**.
3. Click **Enable AI assistant**.

An **AI Assistant** page appears in the navigation. To switch it off again,
return to the same panel and click **Turn off**: the page disappears and the AI
modules stop being imported.

To disable it for everyone on a shared machine, regardless of the setting:

```bash
export SPACEMAP_ASSISTANT=off
```

The panel then shows that an administrator has disabled it.

---

## Choosing where the model runs

| Option | Needs | Privacy |
| --- | --- | --- |
| **Local model** | Disk space, and ideally a GPU | Nothing leaves the machine |
| **Hosted API** | An API key and a network connection | Prompts go to the provider |
| **Neither** | — | The assistant explains what to set up |

The rest of this page covers the local route, which is the default and the one
that keeps data in place.

---

## Installing a local model

### Step 1 — install the AI packages

The panel checks whether `torch`, `transformers`, `accelerate`, `fastapi` and
`uvicorn` can be imported. If any are missing it prints the exact command for
your environment:

```bash
python -m pip install 'spacemap-ai[ai]'
```

On a machine with no GPU, install the GGUF extra instead — it is far smaller
and much faster on CPU:

```bash
python -m pip install 'spacemap-ai[gguf]'
```

Click **Re-check dependencies** afterwards.

### Step 2 — pick a model

The **Model library** lists what SPACEMAP knows about, with its size, its
license, and whether your hardware can run it. The panel reads your actual CPU,
memory and GPU and says so plainly — for example *"No GPU detected, so this
model will run on CPU. Expect slow replies."*

| Model | Size | Needs | Notes |
| --- | --- | --- | --- |
| Qwen3 4B Instruct (Q4_K_M) | ~2.5 GB | No GPU | Text only. The practical choice on a laptop. |
| Gemma 4 12B Instruct | ~24 GB | ~28 GB GPU | Reads images, so it can comment on previews. |
| Llama 3.1 8B Instruct | ~16 GB | ~20 GB GPU | Text only. Access must be approved on Hugging Face. |

Start with **Qwen3 4B** unless you have a large GPU. It needs no GPU, no
license approval and no token.

### Step 3 — gated models need a token

Gemma and Llama require you to accept their terms before downloading. If you
choose one:

1. Sign in to Hugging Face and accept the terms on the model's page (the panel
   links to it).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Paste it into **Hugging Face access token** in the panel.

The token is stored in your configuration directory, readable only by you. You
can also supply it as `HF_TOKEN` instead of saving it.

### Step 4 — download

Click **Download**. The download runs in the background with a progress bar and
survives page changes and browser reloads. When it finishes, click **Use this
model**.

Then open the **AI Assistant** page, go to **Model → Local model**, and start
the server. The first start takes a few minutes while the weights load.

---

## Running without a GPU

This is supported. The server loads the model on the CPU in float32 and answers
correctly, just slowly — minutes rather than seconds for a long reply.

Quantized loading (8-bit and 4-bit) needs a CUDA GPU, because `bitsandbytes`
has no usable CPU build. On a CPU-only machine, use a **GGUF** model instead:
it is quantized ahead of time into a single file and is designed for CPU
inference.

SPACEMAP picks the precision for you:

| Your hardware | What it loads |
| --- | --- |
| GPU with comfortable headroom | bfloat16 |
| GPU with moderate headroom | 8-bit |
| GPU with little headroom | 4-bit |
| No GPU | float32 on CPU |

---

## Installing on a machine with no network

Download the weights elsewhere, copy them across, then use **Install from a
folder on this machine** in the model library. Point it at the folder, and
SPACEMAP moves or copies the files into its models directory and verifies them.

The model server itself never needs the network: it runs with
`HF_HUB_OFFLINE=1` set, so a stale reference can never turn into a silent
download.

---

## Adding a model that is not listed

Use **Add another model**. Give it a short id, a display name, the Hugging Face
repository, and — for GGUF — the file name. Entries are saved to your own
catalog at `~/.spacemap/config/model_catalog.json` and survive upgrades.

The built-in catalog lives in the package and is never modified. A user entry
with the same `id` overrides the built-in one, which is how you correct a
repository name that has changed.

A catalog entry looks like this:

```json
{
  "id": "my-model-8b",
  "name": "My Model 8B Instruct",
  "backend": "transformers",
  "repo_id": "org/my-model-8b",
  "local_dir": "org/my-model-8b",
  "approx_size_gb": 16,
  "min_vram_gb": 20,
  "cpu_capable": true,
  "gated": false
}
```

Set `"backend": "gguf"` and add `"filename": "model-Q4_K_M.gguf"` for a
quantized single-file model.

---

## Moving where models are stored

Model weights are large and often belong on a different disk. Set the models
folder in **Storage locations** in the panel, or:

```bash
export SPACEMAP_MODELS_DIR=/mnt/bigdisk/spacemap-models
```

A read-only shared model store works too: point several installations at the
same folder and each will see the models as installed.

---

## Running the model on a cluster

If SPACEMAP runs on the login node of a Slurm cluster, the assistant can submit
the model server as a job instead of running it on the login node. Enable it in
**Tab 0 → AI Assistant → Run model jobs on a cluster**.

The control is disabled unless `sbatch` is actually on your PATH, so it stays
invisible on a workstation. Partitions are ranked by what Slurm reports — idle
nodes, memory, whether they carry GPUs — and you can name preferred partitions
if your site has conventions.

The assistant never accepts a shell command from chat. It submits only a fixed
resource-probe script that it generated and verified itself, byte for byte,
against the recorded parameters.

---

## What the assistant may touch

By default it may:

- **read** the SPACEMAP source, so it can explain the pipeline;
- **read and write** its own data directory and your projects directory.

It may **never** write to the installed source, the model store, or your
configuration. An administrator can grant additional exact folders, read-only
or read-write, in the **Admin Access** tab. Whole filesystem roots and home
directories are refused.

---

## Privacy

With a **local** model, nothing leaves the machine. The server binds to
loopback only, requires a locally generated key, blocks remote image URLs, and
runs with proxy variables stripped so that localhost traffic — including that
key — can never be routed through an institutional proxy.

With a **hosted API**, your prompts and any attached images go to that
provider. Check your institution's policy before sending anything derived from
human-subject data. The interface says which provider is active on every reply.

---

## Turning it off again

**Tab 0 → AI Assistant → Turn off.** The page disappears immediately.

To also reclaim the disk space:

```bash
rm -rf ~/.spacemap/models
pip uninstall transformers accelerate bitsandbytes    # if you want the packages gone too
```
