# Reusing segmentation training tiles

In **Segmentation → Training**, Place a tile, correct its cells, and Keep tile.
The single **Train** button is beside these actions. **Train on** chooses all
training tiles or selected tiles. Expand **Tiles** to select individual tiles or
all matching tiles; checking a tile switches the scope to Selected. Search limits
select-all; selections outside the search remain selected. An empty Selected
scope disables Train. Validation tiles stay held out. A pending correction tile
must be kept or discarded before training.

A dedicated status directly under the tile/Train row stays orange during launch,
training and scoring, then turns green only after the job completes. It names the
version being trained, follows the job across navigation, and exposes the current
job message on hover. Failure stays red with its reason; cancellation is not
shown as completion. Tile-placement messages cannot replace this job status.

The action saves current Segmentation settings and snapshots selected IDs, the
Training model/session, source paths and file versions. It rereads full-resolution
crops with the current Nucleus/Border channel choices through the same input
builder as segmentation previews, including selected H&E-derived channels. This
does not introduce a new H&E stain-separation algorithm. Missing selected channels
stop the round unless **Allow missing markers / H&E** is selected under Inputs.
Labels come from the existing corrected masks, unchanged. Saved
tile pictures and original predictions are also left intact.

All validation tiles get the same input refresh for the before/after model scores;
they never reach the trainer. The next version continues from the displayed
Training model, independently of the active segmentation model. **Use for
segmentation** still activates the result explicitly. Both scopes always refresh
inputs, including **Fixed markers**. The previous cached-input Train and separate
Retrain selected buttons are replaced by this single workflow. The backend's
legacy training action remains available for older callers.

## Images with fewer markers

Set **Inputs → Allow missing markers / H&E** for panels whose markers vary.
**Fixed markers** remains the default. The optional mode is a tradeoff: it can
improve missing-input performance while reducing full-panel performance. It
applies to either training scope. DAPI must be selected as a
Nucleus input and present in every training/validation/target image. Other selected
markers can be absent. The same corrected mask is paired with four in-memory
input recipes: all available selected channels, without H&E, a reproducible random
subset of the other markers without H&E, and DAPI alone. DAPI is never dropped;
H&E RGB components are omitted together. Source channels are read once per tile;
no augmented TIFFs or duplicate labels are saved. Four input examples per tile
increase training time and memory. The current maximum-projection combination
and nuclear fallback for absent borders match segmentation inference.

Expand **Validation** and choose one model version to see its saved scores.
**Starting model** and **Trained model** columns show AP for each recipe on the
same held-out tiles in that round; hover the headings for the model names.
With no validation tiles it explicitly says performance is unverified.
These scores do not cover every possible marker panel or establish generalization
to unseen tissues. **Original tile predictions → Check saved predictions**
compares the original, possibly older predictions with corrections; it does not
evaluate the latest model. **Model history** is independently collapsed and
searchable. **Score all versions** compares original cached validation inputs,
without refreshing markers or testing missing-marker scenarios. Saved scores
from different rounds may use different tiles/inputs and are not a common ranking.

The model metadata records required DAPI, actual input recipes and missing
optional channels. Studio checks required inputs before moving prior segmentation
results aside, omits unavailable optional channels, and logs the channels used.
Preview reports how many channels were omitted, with their names on hover. This
is a Studio input policy; keep the metadata sidecar with exported model weights.
Old models do not acquire missing-marker training simply by installing this UI.

The strategy is motivated by [modality-dropout research](https://arxiv.org/abs/1908.06683),
but that evidence does not demonstrate performance of this Cellpose model. The
two composed inputs follow [Cellpose's channel interface](https://cellpose.readthedocs.io/en/v3.1.1.1/models.html).

New tiles record their original image path. Older tiles without that field resolve
by exact filename in Segmentation's configured Images folder. If the image has
moved, the action reports the missing path rather than using the saved picture.
Requests are handled by the existing project job lock. Source or mask changes
between submission and reading stop the job; a changed training session cannot
be overwritten by a finishing older round.

Each model has a compact metadata sidecar and, for refreshed rounds, a separate
`.inputs.json` manifest with selected tile identities, actual channels, source
paths, file versions and corrected-mask hashes. No refreshed image files are
written. Training-selection controls do not load image pixels while browsing.

Verification: `tests/test_studio_retraining.py` covers actual TIFF inputs, mask
preservation, selected/validation separation, model continuation, failure paths
and the API request. `tests/viewer_studio_retraining_browser.py` exercises Firefox
controls and submission against a disposable project with a mocked job launcher.
