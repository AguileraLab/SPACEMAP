# Working with this repository

A practical guide for anyone — a colleague or an AI assistant — who needs to
use SPACEMAP on real data and send improvements back to GitHub, without ever
publishing patient data.

Read this before your first commit.

---

## The one rule

**Code goes to GitHub. Data never does.**

Everything SPACEMAP produces while you work — projects, images, CSVs, Parquet
copies, logs, saved labels, interface state — stays on your machine. Only
source code, documentation and tests belong in the repository.

This is enforced, not merely requested. See [How data is kept
out](#how-data-is-kept-out) for what that protection actually covers, and how
to verify it yourself.

---

## Using the published version on your own data

You can point this version at an existing project. Nothing needs converting.

### 1. Install

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP
conda env create -f environment.yml
conda activate spacemap
pip install -e .
```

Full options, including Docker: [installation.md](installation.md).

### 2. Keep your data outside the repository

This matters. Put projects somewhere of their own:

```bash
export SPACEMAP_PROJECTS_DIR=/path/to/your/projects
```

A project directory inside the clone is ignored by Git, so it is safe — but
keeping data outside entirely removes the question.

### 3. Open an existing project

Start the interface:

```bash
spacemap
```

Then **Tab 0 → Load Existing Project**, and choose the folder holding your
project's `<name>.json`. An existing project opens unchanged: the JSON records
directory locations and per-tab settings, and this version reads the same
format.

If the project was created on another machine and its recorded paths no longer
exist, set them again in Tab 0; the rest of the configuration is preserved.

### 4. What this version adds

Nothing about your data changes, but two new files may appear beside large
tables. Both are derived, both are ignored by Git, and both can be deleted at
any time:

| File | What it is |
| --- | --- |
| `<table>.csv.parquet` | A columnar copy, so a screen reading a few columns takes about a second instead of minutes. Rebuilt automatically when the CSV changes. |
| `<table>.csv.labels.parquet` | Labels you assigned but have not yet written into the CSV. |
| `<segmentation>.geojson.spatial.parquet` | A cached spatial index, so opening a batch takes seconds rather than a minute and a half. |

**The labels file is the one to understand.** Saving a label appends to it in
under a second rather than rewriting a multi-gigabyte CSV. SPACEMAP merges
those labels into everything it reads, so the interface always shows them, and
they survive restarting. The CSV itself only changes when you press **Write
Labels Into CSV**.

So: before exporting, sharing, or analysing the CSV outside SPACEMAP, press
that button. Anything reading the CSV directly sees only what has been written
in.

---

## How data is kept out

Three independent layers.

**1. `.gitignore`** blocks projects, data, caches, models, logs and interface
state. Verify any path yourself:

```bash
git check-ignore -v path/to/your/file
```

No output means the file is **not** ignored — stop and ask before committing.

**2. State lives outside the package.** Settings, caches and logs go to
`~/.spacemap`, not into the clone:

```python
from spacemap.userpaths import describe_layout
print(describe_layout())
```

**3. CI blocks machine-specific paths.** `scripts/check_no_local_paths.py`
fails the build if a committed file contains a home directory, a site mount
point or an account identifier. Run it before you push:

```bash
python scripts/check_no_local_paths.py
```

### Check before every push

```bash
git status                 # only source files should be listed
git diff --cached --stat   # what is actually staged
```

If you see anything under `projects/`, a `.csv`, a `.parquet`, a `.geojson`,
an image, or `interface_state.json` — **do not commit**. Tell someone.

---

## Sending a change back

### First time only

```bash
bash scripts/install_hooks.sh
```

This enables the commit template and the check that every commit explains why
it was made.

### Every change

```bash
git checkout main && git pull          # start from what everyone else has
git checkout -b fix/short-description  # never work directly on main

# ... make your change ...

python -m pytest                       # must pass
python -m ruff check src scripts       # must be clean
python scripts/check_no_local_paths.py # must be clean
python scripts/update_ledger.py        # refresh the change ledger

git add -A
git commit                             # template opens; fill in the Why line
git push -u origin fix/short-description
```

Then open a Pull Request on GitHub against `main` and fill in the template.

### Writing the commit message

Every commit must record **why**. That line becomes the "Why" column in
[LEDGER.md](LEDGER.md), and it is the part nobody can reconstruct from the diff
a year later.

```
fix: stop the loader imposing categorical dtypes

load() converted text columns to pandas 'category' to save memory. That broke
assigning a phenotype to an unlabelled cell, which raised "Cannot setitem on a
Categorical with a new category".

Why: a loader that silently changes dtypes is a trap for every caller, and the
memory saving was not worth it.
```

State what changed, then why. Include the measurement if the change is about
performance — "1044 s to 4.6 s on an 8.6M-row table" is worth more than
"faster".

### Pushing on a cluster login node

Git may not prompt for credentials on some nodes. If `git push` returns
"Repository not found" without asking for anything, supply the token directly:

```bash
read -rsp "Token: " GH_TOKEN && echo && export GH_TOKEN
git -c credential.helper='!f() { echo username=YOUR_USERNAME; echo "password=$GH_TOKEN"; }; f' push
unset GH_TOKEN
```

The token needs both the **`repo`** and **`workflow`** scopes — the second
because the repository contains GitHub Actions workflows.

---

## Reporting a problem

Open an issue with:

- the version (`spacemap --version`) and how you installed it
- what you expected, and what happened
- the full error text, not a summary
- for anything slow: the `[MR ...]` trace lines from the terminal, which name
  each step and its duration

Describe the **shape** of a problem file — rows, columns, size — rather than
attaching data from a study.

---

## If you are an AI assistant working on this repository

The points above apply. A few that are easy to get wrong:

**Measure before optimising.** This codebase has had several rounds of work
aimed at the wrong bottleneck because the cost was assumed rather than traced.
Manual Reference prints a timed trace of every stage; use it, and quote real
numbers.

**Do not conflate an isolated benchmark with the user's experience.** "This
function is now 200× faster" is not "the screen is fast". Say which one you
measured.

**Never impose dtypes in a shared loader.** `spacemap.dataio.load` must return
what `pandas.read_csv` would. `tests/test_dataio_dtypes.py` pins this after a
categorical conversion broke labelling at runtime while every test passed.

**Labels must land on the cell they were assigned to.** `labels.py` keys on row
position and refuses to apply labels if the table changed underneath, rather
than risk labelling the wrong cell. `tests/test_label_cell_identity.py` covers
this; do not weaken it.

**Check both source trees if you are working in `SPACEMAP_V17`.** It holds two
copies of the source that have diverged. Change the one the user names, then
copy only the files you touched — never re-sync wholesale.

**Run the full suite before claiming anything works**, and report failures with
their output rather than summarising them.

---

## Where things are

| | |
| --- | --- |
| [README.md](../README.md) | What SPACEMAP is, and the quick start |
| [installation.md](installation.md) | Conda, pip, Docker, and where files live |
| [ai-assistant.md](ai-assistant.md) | The optional assistant and local models |
| [architecture.md](architecture.md) | How the packages fit together |
| [troubleshooting.md](troubleshooting.md) | Common failures and fixes |
| [LEDGER.md](LEDGER.md) | Every change, who made it, and why |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | The Git workflow in plain language |
| [../CHANGELOG.md](../CHANGELOG.md) | What each release means for users |
