## What changed

<!-- One paragraph. What does this do that the code did not do before? -->

## Why

<!-- Required. The reason for the change: the bug it fixes, the need it meets.
     This is the part nobody can reconstruct from the diff later. -->

## How it was tested

<!-- Commands you ran, and what you saw. "pytest passes" is fine for a small
     change; describe manual GUI checks where tests cannot reach. -->

- [ ] `python -m pytest` passes
- [ ] `python -m ruff check src` is clean
- [ ] `python scripts/update_ledger.py` run and the result committed

## Checklist

- [ ] No data, images, model weights or `projects/` folders are committed
- [ ] No absolute paths from my machine appear in any changed file
- [ ] Notebook outputs are stripped
- [ ] The pipeline still starts with the AI assistant switched off
- [ ] Every commit has a `Why:` line
