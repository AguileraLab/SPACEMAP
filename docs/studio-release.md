# Publishing the Studio release

This release prepares **SPACEMAP Studio 0.4.0** as the primary interface while
retaining Classic and reusable analysis modules. It is based on the public
`main` branch, including its academic research license and installation fixes.
The local development branch is not a replacement for public Git history.

## Validation

Validation was performed on Linux with Python 3.10.21 on 2026-10-15.

| Check | Result |
| --- | --- |
| Complete synthetic unit suite | **2,128 passed, 5 skipped**, no failures (8 minutes with four test workers) |
| Browser workflows | 137 checks passed: cell tagging, TLS review/editing, dataset editing, Consensus ML, job timers/logs |
| Wheel and source archive | Build and required Studio/Classic asset checks passed |
| Installed wheel away from source | Project Workspace opens and creates a project with AI off |
| Installed HTTP server | Health and Studio HTML/JavaScript/CSS served on one port |
| Scientific CPU runtime | Compiled PyTorch/torchvision operators and distances against GEOS passed |
| Python requirements | Installed dependency consistency and fresh resolution passed (190 distributions) |
| Frontend | 26 viewer scripts pass syntax checks; Node 20 rebuild reproduces the bundled heatmap |
| Source hygiene | Ruff, Streamlit form checks, path/account/host scan and notebook-output scan passed |
| Credential scan | No credential-shaped strings found in publishable files |

All 328 packaged Python/JavaScript/HTML/CSS runtime files were compared with
the release sources and matched byte for byte. The validation environment used
Streamlit 1.64.0, NumPy 1.26.4, SciPy 1.12.0, pandas 2.3.3, Shapely 2.1.2,
Cellpose 3.1.1.3, PyTorch 2.14.0 and torchvision 0.29.0.

The review corrected job ordering: writing an older job's cached profile must
not move it above later naming jobs. A recorded start time resolves filesystem
timestamp ties. Tests also now exercise the real settings-location resolver,
and the installed-app check targets the page mounted by Studio's server.
UI-test mocks are now restored after use, and worker lifecycle tests allow for
slow scientific imports while retaining their output assertions. The final
run capped OpenBLAS/OpenMP at one thread and Numba at two per test process.

The local environment lacked **OpenSlide's native library** despite a clean
`pip check`. Installed-app checks passed after supplying OpenSlide 3.4.1 from
an isolated Conda environment. The published Conda recipe includes
`openslide<4`; `spacemap doctor` detects this missing dependency. For an existing
environment with that error, activate it and install the native dependency:

```bash
conda install -c conda-forge 'openslide<4'
spacemap doctor
```

The wheel tests reused existing scientific Python packages; a completely fresh
Conda/pip installation remains a CI check. Docker, GPU inference/performance,
optional DeepCell and a full study-data rerun were not validated here. The
GitHub base/AI installation and Docker checks must pass before merging.
Passing synthetic tests does not establish biological accuracy for every
dataset, performance on every GPU, or compatibility with untested drivers.

Python **3.10** is supported by the current dependency stack. Use
`environment.yml` for a new Conda installation. `requirements-lock.txt` records
one Linux environment; it is not a portable installation recipe. AI packages
are optional. Each installation should pass `spacemap doctor` and
`python -m pip check` before analysis.

### GitHub CI follow-up (2026-10-16)

The [first Studio PR run](https://github.com/AguileraLab/SPACEMAP/actions/runs/38957399285)
passed Docker, packaged startup, lint, repository hygiene and frontend checks.
Both test configurations failed the same server-listing test: base had 2,122
passing tests and eight skips; AI had 2,127 passing tests and five skips.

Server discovery incorrectly required its own Python command line to contain
`spacemap` or `streamlit`. GitHub invokes `python -m pytest`, so the test's
registered current process was removed from the list. The local validation
interpreter's path contained `spacemap`, which concealed this assumption.
Launching locally with a plain `python` process name reproduced the failure.

Discovery now recognizes its own live PID without the command-line heuristic;
other PIDs retain the existing identity check. All four focused server tests
pass with the plain launcher, including regressions for retaining the current
server, refusing to stop it, and refusing to stop an unrelated Python process.
Lint and the local-path check also pass. The full GitHub base/AI suites must be
rerun on the corrected branch before merge; the original full-suite and archive
validation above describes the initial release snapshot, not this follow-up.
Rebuild distribution archives from the final merged release before publishing.

## Publish through a pull request

From the prepared `release/studio` checkout:

```bash
git status
git remote -v
git push -u origin release/studio
```

On GitHub, open a pull request from `release/studio` into `main`, titled
**SPACEMAP Studio 0.4.0**. Review the diff and let all CI checks finish,
including clean dependency installation, packaged startup and Docker.
Merge after those checks pass. `main` then becomes the official Studio source;
there is no need to delete the repository or force-push over its history.

After merging, use an up-to-date, clean `main` checkout to publish the tag:

```bash
git switch main
git pull --ff-only origin main
git tag -a v0.4.0 -m "SPACEMAP Studio 0.4.0"
git push origin v0.4.0
```

Create a GitHub Release for `v0.4.0`, use the Studio changelog as the description,
and mark it as the latest release. Move the prepared changelog entry out of
Unreleased and add `date-released` in `CITATION.cff` before final tagging.
Check existing tags first; do not replace a tag that has already been published.

The GitHub release and Python package index are separate. Publishing here does
not upload a new version to PyPI. Rebuild wheel/source archives from the final
tag before distributing them as release assets.

## Privacy and project data

The source release excludes projects, user settings, secrets, caches, weights,
local environments, and 62 local development/benchmark notes. Those notes are
retained locally; the release includes the reviewed user guides instead.
Image assets illustrate synthetic data. Automated
checks cover personal account identifiers, internal storage paths and node
names, including the shipped viewer files. This audit concerns the prepared
release; it does not erase previously published Git history.

Keep study data outside the software checkout. A server should be used on a
trusted network or behind an authenticated access service. Studio's project
lock applies across machines: concurrent experiments need independent project
configurations and output folders.
