# Phenotyping startup and Flow interactions — 22 October 2026

The original full-table profile spent 33.60 seconds building the Phenotyping inputs. It repeatedly stripped millions of identical phenotype labels and parsed large archived job profiles. Step coverage read the same result columns again (3.30 seconds).

Controls now load from schema and image pairings first. Counts and result badges arrive independently; background summaries preserve unsaved folder/column-role input and current image checkboxes. Flow scope uses the same lightweight path and does not wait for archived-job provenance or phenotype coverage.

`studio/summary_cache.py` stores regenerable JSON under the user cache's `studio-summaries-v1` directory. Input identities include role assignments and source, label-journal, and pending-column file stamps. Concurrent requests share a builder. Invalid or unwritable cache files fall back to computation. Result coverage factorizes phenotype names once; job provenance retains only the output names needed by the image list. ROI names are normalized once per summary, rather than in every image/column pair.

A fresh-process run on the same 2,482,061-cell project reused these summaries in 2.68 seconds, versus 33.60 seconds before the change; coverage reuse took 0.044 seconds. The first summary build still reads the source once, in the background. These are profiled summary-call timings, not a promise that an entire Studio session opens in that time.

Focused tests cover source/journal changes, concurrent builders, corrupt cache recovery, fresh-process disk reuse, custom provenance/column roles, and per-image scope. Browser tests deliberately delay summary calculation and verify that controls are already available and unsaved edits remain intact.

Flow timing and cache-management details are in [studio-flow-gating.md](studio-flow-gating.md). Python changes require restarting Studio once. Close the old Flow popup, hard-refresh Studio, and reopen Flow to load the current browser code.

## Model menu follow-up — 23 October 2026

Listing model choices was deserializing the selected model just to obtain
classes and saved-example counts. The menu now reads its manifest only;
prediction and explicit inspection still load the artifact when needed.
Legacy manifests may omit these optional details until model inspection.
Read-only measurements on the same project reduced the model-options call
from 3.001 s to 0.331 s. The input-controls call took 0.312 s after imports.
These are component timings, not end-to-end timings of the user's running
server. A regression test rejects any artifact loading while listing models.
