# Architecture

SPACEMAP has four Python packages under `src/`, separating scientific routines,
interfaces, workflows and optional model services.

```
spacemap_gui  ──────────────┐
  Streamlit interface       │
  10 pipeline tabs          ▼
  optional assistant page  spacemap
                            scientific core
spacemap_assistant ────────►│  (no GUI, no AI)
  optional AI layer         ▲
                            │
spacemap_workflows ─────────┘
  callable step registry
```

| Package | Responsibility | Optional? |
| --- | --- | --- |
| `spacemap` | The science: registration, segmentation, phenotyping, spatial analysis. Importable on its own from a script or notebook. | No |
| `spacemap_gui` | Studio, its viewer/API/workers, and the Classic Streamlit tabs. | No |
| `spacemap_assistant` | The AI assistant: model catalog, installer, runtime, actions. | Model dependencies are optional; package code ships with the base install. |
| `spacemap_workflows` | A registry describing pipeline steps so they can be called programmatically. | No |

Scientific functions can run without opening the GUI or enabling an AI model.
One current coupling is the RESOLVE text parser: `ph_accel_utils` imports the
lightweight parser from `spacemap_gui`. The diagram describes responsibilities,
not a strict enforcement of package boundaries.

## Studio

`spacemap_gui.server` mounts the Classic page and Studio's viewer API on one
port through `st.App`. `viewer/frontend/` contains local JavaScript, styles and
OpenSeadragon; no frontend build is required for the Studio viewer. The Classic
heatmap widget has a separate compiled frontend with bundled dependencies.

`spacemap_gui.studio` validates settings, dispatches commands and starts
background Python workers. Job requests, status and logs live under the
project's `outputs/studio/jobs/`. A shared filesystem lock limits a project to
one active job across servers. Requests capture settings at launch; later UI
changes apply to subsequent jobs. Supported compute workers have an independent
stop watcher and protected result publication.

Viewer sessions and pending table edits are separate from committed analysis
results. Stage-specific workers call the scientific functions and publish
results through checked save paths. User data and project configuration stay
outside the installed Python package.

---

## The pipeline

The following numbers refer to Classic tabs. Studio separates Cell tagging
from Segmentation; its full step mapping is in the README.
Each tab is one stage, and each writes into the project's `outputs/` tree so a
later stage can pick it up.

| Tab | Stage | Core module |
| --- | --- | --- |
| 0 | Project workspace, settings, AI setup | `spacemap_gui.tabs.tab0_project` |
| 1 | Registration (CODEX ↔ H&E, multi-round) | `spacemap.registration` |
| 2 | Background subtraction | `spacemap.background_subtraction_utils` |
| 3 | ROI extraction | `spacemap.utils` |
| 4 | Tissue annotation | `spacemap.tissue_annotation_utils` |
| 5 | Cell segmentation | `spacemap.segmentation_cellpose`, `segmentation_deepcell` |
| 6 | Batch correction | `spacemap.batch_correction_utils` |
| 7 | Phenotyping (Leiden, SOM, threshold, SCIMAP, VORTEX) | `spacemap.ph_*_utils` |
| 8 | Spatial analysis | `spacemap.spatial_analysis_utils` |
| 9 | Visualization and figures | `spacemap.tissue_annotation_figures` |

---

## Where state lives

Application state is written outside the installed package.
`spacemap.userpaths` resolves a root — `$SPACEMAP_HOME`, else
`$XDG_DATA_HOME/spacemap`, else `~/.spacemap` — for settings, caches and models.
Project data, result tables and their sidecars use the project's configured
folders instead. An `XDG_DATA_HOME`
inside `/tmp` is ignored: some clusters set it to `/tmp/$USER` for everyone,
and `/tmp` is local to one node and cleaned.

```
~/.spacemap/
├── config/     assistant.json, model_catalog.json, tokens, server PID record
├── state/      interface_state.json, interface_settings.json
├── cache/      numba, cellpose, huggingface
├── logs/       model-server.log
└── models/     downloaded weights
```

Three layers decide a setting, highest priority first:

1. `<project>.json` — the loaded project's own configuration.
2. `state/interface_state.json` — cross-session directory and tab state.
3. `state/interface_settings.json` — per-directory UI settings.

Earlier versions kept the last two inside the package. They are migrated on
first run.

---

## How the assistant stays optional

Four independent layers, so a failure in any one of them cannot take the
pipeline down:

1. **Settings.** `spacemap.ai_settings.is_enabled()` reads
   `config/assistant.json`, defaulting to `false`. `SPACEMAP_ASSISTANT=off`
   overrides it for the whole installation. The module imports only the
   standard library, so the question can be asked before any AI code loads.

2. **Page registration.** `app.py` only constructs the assistant `st.Page` when
   the assistant is enabled. Studio can still use lightweight action/context
   helpers; opening the pipeline does not require model weights or a model server.

3. **Guarded call sites.** The header launcher and the Tab 0 panel are wrapped
   so that a missing dependency degrades to a caption, never a traceback.

4. **Optional dependencies.** `transformers`, `fastapi` and friends live in the
   `[ai]` extra. `spacemap.ai_settings.missing_local_dependencies()` reports
   what is absent by checking import specs, without importing anything.

CI enforces this: one job installs the package with no extras and asserts the
pipeline imports and the assistant reports itself off.

---

## The model layer

Model choice is data, not code. Adding a model means adding a catalog entry.

```
spacemap.model_catalog      what models exist, where they go on disk
spacemap.hardware           what this machine can run; picks a precision
spacemap_assistant.model_installer   background download, local import, removal
spacemap_assistant.model_runtime     starts and stops the server process
spacemap_assistant.model_service     the server itself (FastAPI, OpenAI-style)
spacemap_assistant.model_client      the loopback client
```

The catalog merges a built-in JSON file shipped in the package with a
user-editable one in the config directory; user entries override built-ins by
`id`.

Two backends speak the same OpenAI-style protocol:

- **`transformers`** — a model directory served by
  `spacemap_assistant.model_service.server`. Loads multimodal models with
  `AutoProcessor`, falls back to `AutoTokenizer` and `AutoModelForCausalLM` for
  text-only models, and falls back to CPU float32 when there is no GPU.
- **`gguf`** — a single quantized file served by `llama_cpp.server`.

The runtime resolves the interpreter (the one running SPACEMAP, unless
`SPACEMAP_MODEL_PYTHON` says otherwise), the port, and the precision, then
launches the server in its own process group and records the PID.

---

## Safety boundaries in the assistant

`spacemap_assistant.paths` is the enforcement point:

- **Readable:** the application root and the installed source, so the
  assistant can explain the code.
- **Writable:** the assistant's own data directory and the projects directory —
  and nothing else, unless an administrator grants an exact folder.
- **Never writable:** the installed source tree, the model store, the
  configuration and state directories, and anything under `.git`.

The local model server binds to loopback only, requires a locally generated
key, accepts no remote image URLs, and exposes no filesystem or command
execution to the model. Proxy variables are stripped from its environment so
that localhost traffic can never be routed off the machine.

The cluster bridge accepts no command from chat. It generates one fixed
resource-probe script, records the parameters that produced it, and before
submitting regenerates the script and compares it byte for byte.
