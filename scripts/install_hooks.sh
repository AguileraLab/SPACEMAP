#!/usr/bin/env bash
# Point this clone at the repository's shared Git hooks and commit template.
#
# Run once after cloning:
#     bash scripts/install_hooks.sh
#
# Hooks are not distributed by Git itself, so every clone has to opt in.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

git config core.hooksPath .githooks
git config commit.template .gitmessage

echo "Hooks enabled:   $(git config core.hooksPath)"
echo "Commit template: $(git config commit.template)"
echo
echo "Every commit now needs a 'Why:' line. It becomes the Why column in"
echo "docs/LEDGER.md. Use 'git commit --no-verify' to bypass in an emergency."
