# Publishing changes to GitHub

A repeatable procedure for reviewing, checking and publishing code changes.
Inspect staged files yourself; ignore rules and automated checks are not a
complete audit of repository contents or history.

If you are an AI assistant: this is the whole job. Do not improvise around it,
and do not skip step 5.

---

## Once per clone

```bash
bash scripts/install_hooks.sh
```

Enables the commit template and the check that every commit records **why** it
was made.

---

## Every change

```bash
cd /path/to/SPACEMAP

# 1  Start from what is already on GitHub
git checkout main && git pull

# 2  Branch. Never work directly on main.
git checkout -b fix/short-description

#    ... make the change ...

# 3  Verify. All four must pass before you go further.
python -m pytest
python -m ruff check src scripts
python scripts/check_no_local_paths.py
python scripts/check_streamlit_forms.py

# 4  Review your changes before staging
git diff --stat

# 5  INSPECT what you are about to commit
git add path/to/changed_file.py path/to/changed_document.md  # use your actual files
git status
git diff --cached --name-only

# 6  Commit (the template opens; fill in the Why line)
git commit

# Record that commit, then save the ledger separately.
python scripts/update_ledger.py
git add docs/LEDGER.md
git commit -m "docs: refresh change ledger" -m "Why: record the reviewed changes."
python scripts/update_ledger.py --check

# 7  Push
git push -u origin fix/short-description
```

Then open a Pull Request against `main` and fill in the template.

---

## Step 5 exists for a reason

A careless `git add -A` once staged **80 files of a user's analysis projects** —
project configs, restore-point autosaves and logs, all carrying absolute paths
to patient-adjacent data. It was caught before pushing. The ignore rule said
`projects/` while SPACEMAP had written to `project/`.

So: read the output of `git diff --cached --name-only`. If it lists any of
these, **stop, unstage, and fix the ignore rule**:

```
project/    projects/    *.csv    *.parquet    *.geojson
*.ome.tiff  *.png of real tissue  interface_state.json
interface_settings.json           anything under logs/
```

Only source code, tests, documentation and configuration belong in a commit.

To check any single path:

```bash
git check-ignore -v path/to/file     # no output means NOT ignored
```

**Never commit while a check is failing.** That is precisely how those 80 files
got in — the path check reported `[FAIL]` and the commit went ahead anyway.

---

## Writing the commit message

Every commit records **what changed** and **why it mattered**. The why becomes
the "Why" column in [LEDGER.md](LEDGER.md) and is the only part nobody can
reconstruct from the diff a year later. A hook rejects commits without it.

```
<type>: <short summary in the imperative, ~72 chars>

What changed, and how, if it is not obvious from the diff. For anything about
performance, give the measurement: "1044 s to 4.6 s on an 8.6M-row table" is
worth more than "faster".

Why: the reason this needed doing -- the bug it fixes, the person it unblocks.
```

Types: `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`, `ci`,
`chore`.

### A real example

```
fix: recognise button keys that begin and end with a placeholder

Opening Manual Reference raised StreamlitValueAssignmentNotAllowedError for
'tools_mr_vv_filter_remove_2'. Streamlit refuses to restore a value for a
button, and the persistence layer was saving one.

A key written f"{key_prefix}_filter_remove_{idx}" has neither a literal prefix
nor suffix, so the scanner that classifies widgets saw nothing to match. The
key was treated as configuration and written into the project file. Literal
segments between placeholders are now collected too; seven such patterns exist,
so six other button families had the same latent fault.

Why: a saved project could make the tab unopenable, and the cause was invisible
because the key never appeared in any list the scanner produced.
```

---

## Describing what changed, for a human

The PR template asks for three things. Keep them short and concrete:

- **What changed** — one paragraph. What does this do that the code did not?
- **Why** — the bug, the need, the measurement.
- **How it was tested** — commands you ran and what you saw. Name the manual
  checks where tests cannot reach, e.g. "opened Tab 7, saved 50 labels,
  confirmed the CSV count went from 102 to 153".

---

## Pushing from a cluster login node

Git often will not prompt for credentials there. If `git push` returns
`Repository not found` **without asking for anything**, it is an
authentication problem, not a missing repository — GitHub returns the same
response for a private repo you are not authenticated to.

```bash
read -rsp "Token: " GH_TOKEN && echo && export GH_TOKEN
git -c credential.helper='!f() { echo username=YOUR_USERNAME; echo "password=$GH_TOKEN"; }; f' push
unset GH_TOKEN
```

`read -rsp` hides the token and keeps it out of shell history.

The token needs **both** scopes:

| Scope | Why |
| --- | --- |
| `repo` | to read and write a private repository |
| `workflow` | because the repository contains GitHub Actions workflows |

If the organisation uses SAML SSO, click **Configure SSO → Authorize** next to
the token on <https://github.com/settings/tokens>. Without it the token works
on personal repositories and returns "not found" on organisation ones.

---

## What CI will check

| Job | Checks |
| --- | --- |
| Lint | `ruff` over `src` and `scripts` |
| Tests | Python 3.10 with and without AI extras |
| Release wheel | archive contents, installed GUI startup and imports without AI packages |
| Frontend | rebuild matches the bundled heatmap assets |
| Hygiene | ledger current; no machine paths; no buttons in forms; notebook outputs stripped |

A red build is a blocked merge. Fix it rather than merging around it.

---

## Releasing a version

Only when a release is intended. Keep the UT Southwestern license in the
release and retain third-party notices. Review the release commit, CI results
and accompanying data-access instructions before making the repository public.

```bash
# bump the version in pyproject.toml and CITATION.cff, update CHANGELOG.md
git commit -m "release: SPACEMAP X.Y.Z" -m "Why: <what this release delivers>."
python scripts/update_ledger.py
git add docs/LEDGER.md
git commit -m "docs: refresh change ledger" -m "Why: record release metadata."
python scripts/update_ledger.py --check
git push

git tag -a vX.Y.Z -m "SPACEMAP X.Y.Z" -m "Why: version cited in <context>."
git push origin vX.Y.Z
```

Then draft a GitHub Release from the tag, pasting that version's `CHANGELOG.md`
section. If Zenodo is connected, publishing the release mints a DOI — cite the
**version** DOI, not the concept DOI, so the reference stays fixed to the code
actually used.
