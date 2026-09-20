# Contributing to SPACEMAP

Thanks for working on SPACEMAP. This page explains how a change gets from your
machine into the published pipeline, and what we ask you to record along the
way.

If you have not used Git before, start with [The five words you need](#the-five-words-you-need).

---

## The five words you need

| Word | What it is | Everyday equivalent |
| --- | --- | --- |
| **Commit** | A permanent snapshot of the project at one moment. | A **save checkpoint** in a game. If something breaks later, you can go back to it. |
| **Branch** | A copy of the project where you build something new without disturbing the working version. | A **parallel universe**. You test ideas safely; the main one stays untouched. |
| **Push** | Sending your saved commits up to GitHub. | **Uploading** your files to cloud storage. |
| **Pull** | Fetching everyone else's newest changes down to your machine. | **Downloading** an update so your copy matches the team's. |
| **Pull Request (PR)** | A proposal asking the maintainers to review your work and merge it. | Submitting a draft for **peer review** before publication. |

The rest of this page is those five words in order.

---

## First-time setup

```bash
git clone https://github.com/AguileraLab/SPACEMAP.git
cd SPACEMAP

# Enable the shared hooks and the commit template. Git does not distribute
# hooks, so every clone has to opt in once.
bash scripts/install_hooks.sh

# Install the package and the development tools.
python -m pip install -e ".[notebook]"
python -m pip install pytest ruff
```

---

## The workflow

### 1. Pull first

Start from what everyone else already has, so your work does not collide with
theirs.

```bash
git checkout main
git pull
```

### 2. Branch

Never build directly on `main`. Make a parallel universe for your change:

```bash
git checkout -b fix/model-server-cpu-start
```

Name it `<type>/<short-description>`, using the same types as commits below.

### 3. Commit, with a reason

Save checkpoints as you go. **Every commit must say why it was made** — that
sentence is what appears in [docs/LEDGER.md](docs/LEDGER.md), and it is the one
thing nobody can reconstruct from the diff a year later.

```
fix: stop the model server refusing to start without a GPU

load_model() raised when torch.cuda.is_available() was false. It now loads
on CPU in float32 instead.

Why: users without a GPU could not use the assistant at all.
```

The `commit-msg` hook rejects a commit with no `Why:` line. That is deliberate.

**Types:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

### 4. Check your work

```bash
python -m pytest            # unit tests
python -m ruff check src    # lint
python scripts/update_ledger.py   # regenerate the ledger
```

CI runs all three. The ledger check fails a PR that forgot to regenerate it.

### 5. Push

Upload your branch to GitHub:

```bash
git push -u origin fix/model-server-cpu-start
```

### 6. Open a Pull Request

Go to the repository on GitHub and open a PR against `main`. Fill in the
template: what changed, **why**, and how you tested it. A maintainer reviews it,
you respond to comments by pushing more commits to the same branch, and once it
is approved it is merged.

---

## The change ledger

[`docs/LEDGER.md`](docs/LEDGER.md) records every change ever made to this
repository: the date, the commit, the author, the reason, and the files
touched. It is generated from the Git history, never edited by hand:

```bash
python scripts/update_ledger.py           # rewrite it
python scripts/update_ledger.py --check   # what CI runs
```

Curated, human-facing release notes live separately in
[`CHANGELOG.md`](CHANGELOG.md). The ledger is the complete record; the
changelog is the readable summary of what each release means for users.

---

## House rules

**Never commit data.** No patient data, no images, no model weights, no
`projects/` folders. `.gitignore` blocks the obvious cases, but it is not a
substitute for checking `git status` before you commit.

**Never commit absolute paths from your machine.** Notebooks build every path
from `DATA_DIR`; code uses `spacemap.userpaths`. A path like
`/home/you/experiments/...` in a committed file is a bug.

**Strip notebook outputs.** Saved outputs bloat the repository and often carry
sample names. Install the filter once:

```bash
python -m pip install nbstripout
nbstripout --install
```

**Keep the AI assistant optional.** The pipeline must run, and every test must
pass, on a machine with no AI dependencies, no model files, no GPU and no
network. If you add assistant code, import it lazily and guard it.

---

## Reporting a problem

Open an issue with the SPACEMAP version (`spacemap --version`), your operating
system, what you expected, what happened, and the full error text. If it
involves a specific image or CSV, describe its shape rather than attaching
patient data.
