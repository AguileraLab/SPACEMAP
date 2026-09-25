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

## Setting it up: one page

Everything is on one page, **AI setup**. It is in Tab 0 (**AI Assistant**) and,
identically, in the **Setup** tab of the assistant page, so you never need to
go to a second place.

**At the top** a row shows what is done (assistant on, packages, model
downloaded, model chosen, model running) and **one button for the next step**:

1. **Packages.** The environment made from `environment.yml` already has them.
   Otherwise the page shows the one command to run, naming SPACEMAP's own
   Python so the packages go into the right environment.
2. **Download** the model suggested for this machine (the largest that fits
   its GPU), or any other from the list below. Progress updates by itself.
3. **Use** it. The first model you download is chosen automatically.
4. **Start** it, or simply ask a question: the chosen model **starts by itself**
   when you ask (about a minute on a GPU), and the chat shows the progress.
5. **Ask a test question** to see it answer.

**Models.** Each model is one row: what it is (size, whether it reads images,
license, whether it fits this machine), its state (not downloaded,
downloading, downloaded, in use: stopped, starting or ready) and its buttons
(Download, Use, Start, Stop, Remove files).

| Model | Size | Needs | Notes |
| --- | --- | --- | --- |
| Gemma 4 12B Instruct | ~24 GB | ~28 GB GPU | Reads images, so it can comment on previews. Checked on an RTX PRO 6000: ready in about 60 s, answers in seconds. |
| Llama 3.1 8B Instruct | ~16 GB | ~20 GB GPU | Text only. Access must be approved on Hugging Face. |
| Qwen3 4B Instruct (Q4_K_M) | ~2.5 GB | No GPU | Text only; needs `pip install -e ".[gguf]"`. |

**Find more models on Hugging Face.** Type a few words (for example
*qwen3 instruct* or *medgemma*), choose text or text-and-images, and press
**Search**. Results show downloads, size and license; **Add** puts a model in
the list, where it is downloaded and used like the others. Only models with
`safetensors` weights are offered, since those are the ones SPACEMAP's model
server loads.

**Gated models** (Gemma, Llama) need their terms accepted on Hugging Face and a
read token (<https://huggingface.co/settings/tokens>), saved under **Storage
and access**. It is stored in your configuration folder, readable only by you;
`HF_TOKEN` works too.

**Storage and access** shows the models folder and its free space. On a
cluster choose a folder on work storage: home folders are often small.

**Answer with an online model instead (optional)** chooses the NVIDIA hosted
model and its key. It is off unless you choose it; your question and recent
chat then leave the machine.

**Advanced** holds the GPU memory mode (auto, full quality, 8-bit, 4-bit),
answer settings, cluster (Slurm) settings, folder access for AI tools, the
activity log, the model server's log, and **Turn the assistant off**.

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
**AI setup → Advanced → Cluster**.

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
or read-write, in **AI setup → Advanced → Folder access for AI tools**. Whole filesystem roots and home
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

**AI setup → Advanced → Turn the assistant off.** The page disappears immediately.

To also reclaim the disk space:

```bash
rm -rf ~/.spacemap/models
pip uninstall transformers accelerate bitsandbytes    # if you want the packages gone too
```
