# A shared, read-only SPACEMAP installation

One shared code checkout can serve several users. Each user runs a separate
SPACEMAP process in their own Python environment and compute allocation. This
is a shared installation, not a single application server shared by everyone.

## What belongs where

| Location | Contents | Who can write |
| --- | --- | --- |
| Shared release checkout | Source, packaged defaults, documentation | Release maintainer only |
| Each user's project folders | Images or image links, project JSON, annotations, tables, analysis results, review records | That user and explicitly permitted collaborators |
| Each user's `SPACEMAP_HOME` | Personal settings, project registry, caches, logs, downloaded models, optional assistant data | That user |
| Each user's Python environment | Dependencies and installed packages | That user |

Project results belong in the selected project/output folders. Some settings
and caches are application-wide, so they need a personal folder outside both
the shared code and individual projects. Do not give different users the same
`SPACEMAP_HOME`. Do not share the owner's admin Studio URL with other users as
their personal workspace; scoped Review links are a separate feature.

## Maintainer setup

Ask the cluster administrator for a group-readable installation directory.
Only the maintainer should have write permission. Other users need read access
to files and traversal access to directories, including all parent folders.
Any inherited ACLs must also deny them write access. Making a Git checkout
clean does not enforce these permissions; the filesystem does.

Use a reviewed, merged GitHub revision. Keep projects and generated files out
of this checkout. Record its commit with `git rev-parse HEAD` and provide the
same compatible dependency specification to users. Updating code can change
dependencies, so communicate release updates and have users update their
environments as needed. Do updates while users' applications are stopped, or
publish a separate versioned directory and switch users between sessions.

## User setup

Create a personal Python 3.10 environment following
[installation](installation.md), including native libraries and the appropriate
CPU/GPU packages. Install from a maintainer-provided release wheel, or from a
personal clone at the same release revision. Do not run an editable install or
package build inside the read-only shared checkout.

With that environment active, use this launch example. Replace the shared
folder and, if needed, the personal storage locations:

```bash
export SPACEMAP_RELEASE=/shared/apps/SPACEMAP_v18
export SPACEMAP_HOME="$HOME/.spacemap-v18"
export SPACEMAP_PROJECTS_DIR="$HOME/spacemap-projects"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SPACEMAP_HOME/cache"
export NUMBA_CACHE_DIR="$SPACEMAP_HOME/cache/numba"
export MPLCONFIGDIR="$SPACEMAP_HOME/cache/matplotlib"
export HF_HOME="$SPACEMAP_HOME/models/huggingface"
export TORCH_HOME="$SPACEMAP_HOME/models/torch"
export CELLPOSE_LOCAL_MODELS_PATH="$SPACEMAP_HOME/models/cellpose"
mkdir -p "$SPACEMAP_HOME" "$SPACEMAP_PROJECTS_DIR" \
  "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" \
  "$CELLPOSE_LOCAL_MODELS_PATH"
export PYTHONPATH="$SPACEMAP_RELEASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$SPACEMAP_PROJECTS_DIR"
python -m spacemap_gui.launch --server.address=127.0.0.1 --server.headless=true --port 8502
```

Use persistent personal work storage instead of `$HOME` for large projects or
models when your cluster has a small home quota. Previously configured
`SPACEMAP_CONFIG_DIR`, `SPACEMAP_STATE_DIR`, `SPACEMAP_CACHE_DIR`,
`SPACEMAP_LOGS_DIR`, `SPACEMAP_MODELS_DIR`, and `SPACEMAP_ASSISTANT_DATA_DIR`
override the corresponding locations; check that these also point to personal
storage. Project permissions, not this launcher, decide who can edit data.

The shared `src` directory takes precedence over another SPACEMAP installation
in the environment. Python bytecode and common library caches stay out of the
shared checkout. Start in a personal directory so relative output paths also
resolve outside the shared code.

Connect using your cluster's SSH forwarding or approved internal access method.
The launcher selects another free port if the requested port is busy; use the
address it prints. In **Project**, open or create your own project, and select
output folders you can write. Your jobs consume your allocation and your
outputs consume the quota of the folders you choose, rather than the release
maintainer's quota merely because you use their code.

## Request to cluster IT

> Please provide read/traverse access to the approved SPACEMAP release directory
> for our user group, with write access restricted to the release maintainer.
> Check parent-directory permissions and inherited ACLs. Each user will run a
> separate application under their own account and compute allocation, with
> projects, settings and caches stored in their own writable locations. We do
> not need group write access to the application source or packaged JSON files.

This setup does not provide a permanent reviewer URL across changing compute
nodes. For that separate requirement, see [internal review hosting](studio-review-vpn.md)
or [HTTPS review hosting](studio-review-https.md).
