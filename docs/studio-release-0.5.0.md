# Studio 0.5.0 release candidate

This release combines the new Studio workflows with the installation and
portability fixes already published in 0.4.0. It is prepared on a release
branch; it is not published until that branch is reviewed and merged.

## Included

- Review queues and scoped sharing, reviewer suggestions, owner approval and
  separate reviewer display preferences.
- Gland-composition types and exterior bed niches, K selection and silhouette
  diagnostics. These describe measured compositions, not genetic clonality.
- Spatial feature screening, distance zones, sample-level comparisons, explicit
  multiple-testing families, resampling and separate per-image input tables.
- Flow gating and figure layouts, marker review and reusable phenotype models.
- Updated annotation, cell-layer, selection, display and background-job workflows.

See the dedicated [gland guide](studio-gland-types-guide.md),
[screening guide](studio-feature-screening-guide.md),
[Flow guide](studio-flow-gating.md), [marker review](studio-marker-review.md),
[internal review hosting](studio-review-vpn.md) and
[HTTPS review hosting](studio-review-https.md).

## Installation

Use Python 3.10 and the [installation guide](installation.md). Optional Flow
transforms and vector/PDF export use `pip install ".[flow]"`. Study data, model
weights, user settings and reviewer credentials are not part of the release.
The existing academic research license remains unchanged. For a shared cluster
installation, use the [read-only installation guide](shared-installation.md).

## Validation

Release-candidate checks on Linux / Python 3.10:

- The full pytest run completed with 2,635 passes, six skips and seven failures.
  All seven failures passed in targeted reruns after updating stale expectations
  and fixing script/style cache-version handling. The entire suite was not
  rerun after these final corrections.
- Real Firefox checks covered review/sharing, display persistence, cells,
  gland types, spatial screening and Flow gating with disposable fixtures.
  The reviewer layout regression check passed 43 assertions, including navigation,
  annotation controls and isolation from the owner's display preferences.
- Frontend rebuild reproducibility, JavaScript syntax, lint, local-path checks,
  wheel/sdist assets, installed application startup and CPU runtime checks passed.

Installation checks reused the available scientific dependencies; they do not
establish that a fresh dependency resolution will work on every machine. A new
GPU benchmark, Cellpose model-weight inference, Docker build and other operating
systems were not tested for this candidate. Run the relevant installation checks
and CI before publishing. Synthetic software tests do not validate biological
findings.

## Keep a clean GitHub mirror

Develop and analyse data outside the release checkout. While preparing a release,
this folder contains a proposed update on `release/studio-0.5.0`, not an exact
mirror of public `main`. Review and commit only the prepared source changes,
then push that branch and open a pull request. Follow
[publishing changes](publishing-changes.md), including the staged-file review
and change-ledger update.

After the pull request is merged and the working tree is clean:

```bash
git switch main
git pull --ff-only origin main
git status --short
```

No output from the last command means the checkout has no local changes.
Use `git fetch origin` and compare `git rev-parse HEAD` with
`git rev-parse origin/main` to confirm that it matches the remote revision.
A fixed folder name does not automatically update the code: pull explicitly.

Keep projects and application state outside this folder. New work belongs on a
separate development branch or worktree; do not run research analyses into this
release checkout. Rebuild release archives from the final merged commit before
publishing them as release assets.
