# Working in SPACEMAP Studio

Start `spacemap`, create or open a project, and click **SPACEMAP Studio** in the
top bar. Studio combines the pipeline steps on the left, an image in the middle,
and Layers, Jobs and other tools on the right. The classic tabs remain available
through **Classic SPACEMAP** and the step's **Open Tab** button.

## Images, markers and annotations

Choose an image from a step's work list. In Layers, select marker channels and
adjust their display ranges. Auto controls change visualization contrast; they
do not change phenotype labels. Review nuclear and border markers before
segmentation. MPP means micrometres per full-resolution image pixel.

Annotation tools draw and edit shapes on the image. Imported GeoJSON coordinates
must use that same image coordinate system. Save/review boundaries before using
them for cell tagging or spatial interpretation.

## Cell tables and manual reference

The Data panel below the viewer links rows to the image. Filter or browse the
dataset, edit values and column names, and use the available save/undo controls.
Check the edit's scope: visible/selected cells and the whole dataset are different
operations. Export or save the corrected table before sharing it.

Manual-reference labels and checked cells can train suggestion models. MR-ML
can use a trained reference model or reference-weighted method agreement.
Consensus ML learns from cells where the selected phenotyping methods agree.
Validation scores measure agreement with the chosen reference or held-out agreed
cells; they do not independently establish biological truth.

## Jobs and computing

A run snapshots its settings when it starts. Changing CPU cores or other controls
then affects a new run, not a worker already running. Jobs retain logs and saved
results. Elapsed time updates independently of the log.

Cell tagging reports loading, annotation preparation, tissue assignment,
distance calculations, bed assignment and saving. For larger gap-aware searches
it reports actual completed-cell milestones. Its progress bar counts stages and
images, not estimated remaining runtime. Gap-aware geometry uses CPUs; the GPU
option applies to supported ordinary boundary-distance calculations.

Supported compute workers can be interrupted by Stop. Publication of completed
outputs is protected; stopping during that short phase can wait for the write
to finish. Do not delete locks while a worker may still be writing.

One active job is allowed per project across shared storage. To compare runs on
another machine, create an independent project with separate configuration,
outputs and job history, and point its inputs at the same images/tables. Copy
annotations if they might be edited during the comparison. An output-folder
change alone does not create an independent project.

## Spatial interpretation

A cellular neighbourhood describes the phenotype composition around a cell.
Two distant cells can have the same neighbourhood label because their local
compositions are similar; the label does not define one connected structure.
Check the neighbour count, grouping boundaries and scaling when comparing runs.

**TLS candidates** detect and summarize connected lymphoid aggregates using the
chosen phenotypes and spatial parameters. Review their boundaries, B/T-cell
organization and supporting markers. Candidate status alone does not establish
TLS maturity or diagnostic validity. Cell membership, structure ID, area and
signed distance describe the selected boundary: distance is negative inside,
positive outside and zero at the boundary.

Keep calibration, annotation versions, model versions and analysis settings with
your results. A successful job or a clean software test suite is not a substitute
for image and biological review.
