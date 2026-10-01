# SPACEMAP handover

Everything a new person or AI assistant needs to work on SPACEMAP without
rediscovering it. Read this first; it will save you a day.

Companion documents: [architecture.md](architecture.md) for how the packages
fit together, [working-with-this-repository.md](working-with-this-repository.md)
for keeping data out of Git, and [LEDGER.md](LEDGER.md) for every change ever
made and why.

---

## 1. What SPACEMAP is

An end-to-end pipeline for **spatial multiplex imaging** — multi-channel
microscopy of tissue where each channel is a protein marker. It takes raw
acquisitions through to spatial statistics and publication figures.

Built by the Aguilera Lab at UT Southwestern. The accompanying manuscript,
*"Resolving phenotyping discordance with SPACEMAP, an integrated machine
learning framework"*, is on bioRxiv (`10.1101/2025.11.27.690631`) and was
submitted to Nature Communications.

### The pipeline, tab by tab

| Tab | Stage | What it does |
| --- | --- | --- |
| 0 | Project workspace | Create/load a project; configure directories and the optional AI assistant |
| 1 | Registration | Align CODEX rounds to each other and to H&E |
| 2 | Background subtraction | Remove autofluorescence |
| 3 | ROI extraction | Cut regions from whole-slide images |
| 4 | Annotation | Tissue compartments and zones |
| 5 | Segmentation | Cellpose, or DeepCell/Mesmer |
| 6 | Batch correction | Harmonise marker intensities (ComBat) |
| 7 | Phenotyping | Leiden, SOM, thresholds, SCIMAP, VORTEX, ML consensus, **Manual Reference** |
| 8 | Spatial analysis | Neighbourhoods, distances, interactions, tissue continuity |
| 9 | Visualization | Proportions, heatmaps, pairwise comparisons, figures |

Each stage writes into the project's `outputs/` tree, so stages compose and you
can stop and resume anywhere.

### The four packages

```
spacemap            the science. No GUI, no AI. Importable from a script.
spacemap_gui        the Streamlit interface and its tabs.
spacemap_assistant  the optional AI assistant.        <- removable
spacemap_workflows  a registry describing pipeline steps.
```

`spacemap` never imports the other three. That is what lets the pipeline run
headless and the assistant be absent entirely.

### Scale you must design for

This is the single most important fact about the codebase:

- unified per-cell tables: **13 GB, 8.6 million rows, 133 columns**
- OME-TIFF images: **32 GB, 38 channels, 56000 × 34500 pixels, 256×256 tiles**
- segmentation GeoJSON: **891 MB, 1.4 million cell polygons**

Anything that touches all rows, all columns, or all polygons will take minutes.
Assume nothing is cheap.

---

## 2. What was done, and why

Two bodies of work. Both are complete and committed; nothing is pushed.

### A. Preparing for public release

The pipeline only ran from one lab's checkout on one cluster. Made installable
by anyone:

- **`spacemap.userpaths`** — all writable state resolves to `~/.spacemap`
  (or `$SPACEMAP_HOME`), never inside the installed package. Previously the
  interface wrote `interface_state.json` *into* `src/`, carrying one
  developer's absolute paths to every user.
- **The AI assistant is optional and off by default.** Its page is not
  registered and its modules are not imported until switched on in Tab 0.
  `SPACEMAP_ASSISTANT=off` disables it installation-wide.
- **`spacemap.model_catalog`** — model choice is data, not code. A built-in
  JSON catalog plus a user-editable override; models install from the GUI,
  downloaded from Hugging Face or imported from a folder offline.
- **The local model server runs on CPU.** It previously refused to start
  without CUDA.
- **`cluster.py`** provides portable Slurm support with no site names,
  hidden unless `sbatch` exists.
- `deepcell` moved out of the base install (it pulls TensorFlow).
- Packaging, Docker, CI, license and citation metadata, contributor docs.

**A dependency fix worth knowing:** `pip install .` was impossible.
`palom` requires `zarr>=2.10,<2.15` and `numpy<2.0`, while the project asked for
`zarr>=2.16` and `numpy<2.3`. **palom is the binding constraint on this stack**
— raising either pin without raising palom makes the dependency set unsolvable.

### B. Making it usable on cohort-sized data

Opening Spatial Analysis or Visualization took minutes; Manual Reference took
thirteen. Four separate causes, all measured:

| Cause | Before | After |
| --- | --- | --- |
| Reading all 133 columns for a 2-column question | 264 s | **1.1 s** |
| `st.cache_data` unpickling a 15 GB frame per click | 20–30 s | **0** |
| Normalising 8.6M rows to display 50 cells | 1044 s | **4.6 s** |
| Building 1.4M shapely polygons per batch | 664 s | **89 s** |

Manual Reference end to end: **13 minutes → 37 seconds**.

**`spacemap.dataio`** is the single place the GUI asks for data:

```python
dataio.columns(path)                 # names only, reads no rows
dataio.load(path, columns=[...])     # only those columns
dataio.start_fast_copy(path)         # build the Parquet twin, in background
```

It keeps a **Parquet twin** beside each large CSV, built in a background thread
on first use and rebuilt whenever the CSV changes, validated by a size+mtime
fingerprint in the Parquet metadata. Every read falls back to the CSV if the
twin is missing, stale or still building.

> **Why a twin, and not just `usecols`.** A CSV is one run of text, so reaching
> column 68 of row 5,000,000 means parsing everything before it. `usecols`
> saves memory but almost no time — 264 s becomes 72 s. Parquet stores columns
> separately, so unwanted columns are never read: 1.1 s. **Column selection
> alone does not work on a CSV.**

**`spacemap.labels`** stores labels assigned in the interface. Saving one used
to rewrite the whole 13 GB table (~13 min) and invalidate the Parquet twin
(~2.7 min more). Labels now append to a sidecar in under a second and are
merged into everything SPACEMAP reads. The CSV changes only when the user
presses **Write Labels Into CSV**.

> **The sidecar keys on row position**, so it records the table's size and
> mtime and **refuses** to apply labels if the table changed underneath,
> rather than labelling whatever cell now sits at that position.
> `tests/test_label_cell_identity.py` guards this. Do not weaken it.

### C. Bugs found and fixed along the way

| Bug | Effect |
| --- | --- |
| `generate_qc_plots_emd_silhouette` had 18 lines of another function spliced in | Raised `NameError` every call; the silhouette figure was never saved |
| Filtered-CSV export referenced `filter_rows` | `NameError` in the audit trail |
| Cellpose PNG preview used `pyplot` without importing it | Preview silently never worked |
| `load_unified_csv.clear()` on a plain function | Reload buttons did nothing |
| Button keys beginning *and* ending with a placeholder | Persisted as settings; a saved project made Tab 7 **unopenable** |
| Top-5 markers ranked by Z1 | Rare markers outranked strong broad ones — see §4 |

---

## 3. Hard-won lessons

Read these before changing anything. Each cost real time.

**Measure before optimising.** Two rounds of work targeted the wrong
bottleneck because the cost was assumed. Manual Reference now prints a timed
trace of every stage — use it. `SPACEMAP_TRACE=0` silences it.

```
2026-09-22 17:15:49 | Tab 7 Phenotyping | Manual Reference | Started: load columns   n=27
2026-09-22 17:15:52 | Tab 7 Phenotyping | Manual Reference | Finished: load columns (3.0 s)
```

Every tab prints in this format through `spacemap_gui/terminal.py`; use
`terminal.say(tab, message, step=...)` and `terminal.timed(...)` rather than
`print`.

**Run log.** `spacemap_gui/run_log.py` records every action around `pg.run()`
in `app.py` into `<project>/logs/run_log.csv` (time, CPU, peak RAM and GPU,
settings changed). A tab needs no wiring; a job in its own process calls
`run_log.record_run` itself, as the Leiden worker does. There is one switch per
tab, `tabN_create_log`; do not add another log checkbox.

**An isolated benchmark is not the user's experience.** "This function is 200×
faster" is not "the screen is fast". Say which you measured.

**Never impose dtypes in a shared loader.** `dataio.load` must return what
`pandas.read_csv` would. Converting text columns to `category` to save memory
broke labelling at runtime — "Cannot setitem on a Categorical with a new
category" — while all 748 tests passed. `tests/test_dataio_dtypes.py` pins it.

**`st.cache_data` unpickles on every hit.** It is for small results only. A
full frame must bypass it and live in `session_state`.

**`st.button` is illegal inside `st.form`.** The test suite never renders these
screens, so it reaches the user as a traceback. `scripts/check_streamlit_forms.py`
catches it.

**A setting is saved only if its control has a tab-prefixed key.**
`persistence.sync_project_state` saves whatever is in `session_state` under a
key that `get_tab_affected_from_key` maps to a tab and `is_transient_key` does
not reject. A control without `key=`, or with a prefix no tab owns (Tab 3's old
`tma_*`), is silently lost on reload. Seed keyed controls with
`ui_utils.seed_widget_state` and omit `value=`/`default=`.
`tests/test_widget_keys_persist.py` fails on any literal key that would be lost.

**Inspect what you stage.** A careless `git add -A` committed 80 files of a
user's analysis projects, because `.gitignore` said `projects/` while SPACEMAP
wrote to `project/`. Both are ignored now, and
`scripts/check_no_local_paths.py` inspects what Git would commit.

**`SPACEMAP_V17` holds two diverged source trees.** `SPACEMAP_AI/src` is the
active one; each launcher puts its own `src` on `PYTHONPATH`. Change the tree
the user names, then copy **only the files you touched** — never re-sync
wholesale.

**Git does not prompt for credentials on these cluster nodes.** See §5.

---

## 4. Open questions and unfinished work

**Top-5 marker ranking (needs scientific judgement).** Ranking is decided
entirely by Z1: Z2, CDF and NLog are all order-preserving within a cell. Z1
only compares like with like when distributions have similar shape, and they do
not — FoxP3 had median 2.2 and max 2165 while CD4 was far broader, so a cell at
the 99.0th percentile for FoxP3 outranked one at the 99.5th for CD4. Percentile
ranking is implemented and is now the default, with the old behaviour available.
**It has not been validated against images by a human.** Early comparison showed
CD20 ranking first on several T-cell-looking cells, which may point at
segmentation spillover or a corrected-vs-raw column question. Unresolved.

**Progressive rendering.** Manual Reference's remaining ~31 s is drawing 50 cell
images at ~0.6 s each. Showing them one at a time would let labelling start
after a second. The loop collects into `results_to_store` before rendering.

**Resolve's Z1 score mode is a relative rule.** Positive means "Z1 above 1.5
within the batch" (`ph_thres_utils.apply_z1_positivity`). Abundant markers and
markers with a few very bright cells are under-called. It is a fallback for
projects without threshold tables; it has not been validated against images.

**Saved-but-unwritten columns.** Phenotype saves go to
`spacemap.pending_columns` (whole columns beside the CSV, keyed by table
fingerprint, with the cell IDs kept for re-applying) and reach the CSV only
when the user presses "Write pending changes into the CSV". Anything that
reads the CSV directly -- notebooks, `pd.read_csv` in older code -- does not
see them until then; the file panel says so. Reads through `dataio` do.

**Why the Leiden worker still reads the CSV with pandas.** pyarrow (which
`dataio` and the Parquet copy use) differs from pandas in the last binary digit
for about 1 value in 10 million, and types an all-blank column as text where
pandas reads float. Switching the worker would save about 2.5 of its 22 minutes
but could shift cluster assignments against earlier runs. It is a decision for
the lab, not a silent change.

**Tab 9 `heatmap.py` still reads broadly**, deliberately — it scans many columns
for low-cardinality filter columns. Narrowing it needs care.

**Publication status.** The GitHub repository is private. The authors plan to
make it accessible when the manuscript is submitted. Confirm the reviewed code
revision and its license files are on the published branch before release.

**Licence.** SPACEMAP uses the UT Southwestern academic-research-only terms in
[LICENSE](../LICENSE). Commercial use and redistribution, including any use or
redistribution by a for-profit entity, are prohibited. The
[UT Southwestern release policy](https://www.utsw.edu/about-us/administrative-offices/technology-development/agreements/open-source-release-of-software.html)
requires express written permission from the Office for Technology Development
for a different license. Third-party dependencies and model weights retain their
own terms.

---

## 5. Publishing your changes to GitHub

The full contributor workflow is in
[../CONTRIBUTING.md](../CONTRIBUTING.md). This is the short, repeatable version
an assistant can follow.

### Every change, start to finish

```bash
cd /path/to/SPACEMAP

# 1. Start from what is already there.
git checkout main && git pull

# 2. Work on a branch, never directly on main.
git checkout -b fix/short-description

# ... make the change ...

# 3. Verify. All four must pass.
python -m pytest
python -m ruff check src scripts
python scripts/check_no_local_paths.py
python scripts/check_streamlit_forms.py

# 4. Review the changed files.
git diff --stat

# 5. Inspect what you are about to commit. Do not skip this.
git add path/to/changed_file.py path/to/changed_document.md  # use your actual files
git status
git diff --cached --name-only

# 6. Commit, with a Why line.
git commit

# Record that commit, then save the ledger separately.
python scripts/update_ledger.py
git add docs/LEDGER.md
git commit -m "docs: refresh change ledger" -m "Why: record the reviewed changes."
python scripts/update_ledger.py --check

# 7. Push.
git push -u origin fix/short-description
```

Then open a Pull Request against `main` on GitHub.

### Step 5 is not optional

If `git diff --cached --name-only` lists anything under `project/`,
`projects/`, a `.csv`, `.parquet`, `.geojson`, an image, or
`interface_state.json` — **stop**. That is patient-adjacent data. Unstage it and
fix the ignore rule.

### Writing the commit message

Every commit must record **why**. That line becomes the "Why" column in
[LEDGER.md](LEDGER.md), and it is the only part nobody can reconstruct from the
diff later. A `commit-msg` hook rejects commits without one; enable it once per
clone with `bash scripts/install_hooks.sh`.

Structure: **what changed**, then **why it mattered**. Include the measurement
for anything about performance.

```
perf: normalize the cells shown, not the whole table

Manual Reference normalized every row before selecting the few dozen cells to
display. On an 8.6M-row table: loading 3 s, to_numeric 86 s, normalizing
1044 s -- for values that take 16 ms on the cells actually shown.

Only the Z1 step looks across cells; Z2, the CDF and the negative-log steps are
per-cell or per-value. The population pass now runs once and the per-cell
transform is applied to each candidate as the loop reaches it. Verified
identical across all six normalization configurations; largest difference
5.8e-14.

Why: a reviewer waited thirteen minutes to label fifty cells, and almost all of
it was arithmetic on cells that were never displayed.
```

Good subject prefixes: `feat`, `fix`, `perf`, `docs`, `refactor`, `test`,
`build`, `ci`, `chore`.

### Pushing from a cluster login node

Git may not prompt for credentials there. If `git push` returns "Repository not
found" without asking for anything, supply the token directly:

```bash
read -rsp "Token: " GH_TOKEN && echo && export GH_TOKEN
git -c credential.helper='!f() { echo username=YOUR_USERNAME; echo "password=$GH_TOKEN"; }; f' push
unset GH_TOKEN
```

The token needs **both** the `repo` and `workflow` scopes — the second because
the repository contains GitHub Actions workflows. If the organisation uses SAML
SSO, click **Configure SSO → Authorize** next to the token.

### What CI checks

Lint; tests on Python 3.10 with and without AI extras; an install with **no extras** proving the
pipeline works without the AI packages; the ledger being current; no
machine-specific paths; no buttons inside Streamlit forms; notebook outputs
stripped.

---

## 6. Quick reference

```bash
spacemap                       # start the interface
SPACEMAP_HOME=/path spacemap   # move all writable state
SPACEMAP_ASSISTANT=off         # disable the AI assistant
SPACEMAP_FAST_COPY=off         # disable Parquet twins
SPACEMAP_TRACE=0               # silence the Manual Reference trace
SPACEMAP_PROJECTS_DIR=/path    # default location for new projects
```

```python
from spacemap.userpaths import describe_layout; describe_layout()
from spacemap import dataio;  dataio.load(path, columns=["phenotype_leiden"])
from spacemap import labels;  labels.pending_count(path)
```

Test baseline: **786 passed, 13 skipped** in the published repo (2026-09-22);
**760 passed, 2 skipped** in `SPACEMAP_AI` (which has assistant tests the
published repo skips without the AI extras).
