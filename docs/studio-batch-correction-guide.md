# Batch correction in Studio

## Preparing a new run

1. In **Folders and tables**, verify Images, Cell tables and Output. Use raw
   measured marker columns from Segmentation or Cell tagging, not already
   transformed or corrected columns. Tick the images to include.
2. Set **Output name**, the base filename. Studio adds a date and unique run
   suffix. Parquet is always saved; **Save CSV too** creates the same table as CSV
   beside it. Jobs → Saved results can create or update a CSV copy later.
3. Expand **Samples**, immediately below the list. This is an editable metadata
   table: assign your sample information to the
   correct ROIs or cell-level sample IDs. It is not a correction-results summary.
4. In **Settings**, choose the technical batch and markers. Use slide only if it
   represents the staining/acquisition batch. Under **Keep biological differences**,
   use the examples below to choose which boxes to tick.
5. Expand **Cell filters**. Set the filter column/value, area column and percentile
   range, marker-outlier percentiles, and optional corrected indicator. These
   autosave into the same configuration used by classic. Review Post-correction
   QC before enabling exclusions.
6. Click **Correct: Selected**. Inspect Run notes, Distribution checks and known
   biological populations. Choose **Use this table in Studio** to activate the new
   saved result downstream.

**Combined table** summarizes the active saved result, independently of settings
for the next run. A changed-input notice describes that saved result's recorded
dependencies; it does not block a new correction. Existing outputs are retained.

## What should I tick?

**Tick a sample characteristic when you want to keep its differences and the
groups share slides.** These examples assume **Correct between = slide**.
For another correction unit, apply the same rule to that unit.

| Example | Tick? |
|---|---|
| Each slide contains T1 and T2 samples, repeated across slides. | **Yes** for the timepoint column. |
| T1 is only on one set of slides; T2 only on a different set. | **No**: the timepoint difference cannot be separated from the slide effect. |
| Groups A and B share slides. | **Yes** for the group column, if that difference matters to your analysis. |
| A appears on slides 1–6 and B on slides 7–12. | **No**: several slides per group is not enough when the groups never share slides. |
| Each patient appears on only one slide. | **No** for patient identity, even if several patients share each slide. |
| Patients A and B both have samples on slides 1 and 2. | **Yes** for patient identity if you want to keep differences between patients. |
| A patient has samples on several slides, but those slides contain no other patients. | **No**: separate sets of slides per patient still cannot separate patient from slide. |

For example, with one patient per slide and T1/T2 samples from that patient on
every slide, tick **timepoint**, not **patient**.

Not every group needs to appear on every slide. Shared groups can connect the
slides: A/B on slide 1 and B/C on slide 2 can separate the group and slide effects.
This establishes whether the effects can be separated; enough cells are also
needed within the cell populations being corrected.

Leaving an unsuitable box unticked does **not** solve the study-design problem:
correction may remove real group differences. This limitation is also checked
in [ComBat's implementation](https://github.com/jtleek/sva-devel/blob/master/R/ComBat.R).

Tick only relevant differences. Studio combines multiple checked columns, so
the combinations must also be separable from slides. Review **Run notes** for
cell groups where protection could not be used. The column selected in
**Correct between** belongs off here. All unchecked sample columns are still saved.

## What filters do

Filters operate in this order: column/value, area percentiles, then marker
outliers. Percentiles are pooled across the selected cells remaining at each
step. A marker outlier exceeds the upper threshold in more than half the selected
markers, or falls below the lower threshold in more than half. Setting 0–100
disables percentile exclusions; correction still requires finite marker values.

As in classic, the column/value filter keeps tables that lack that column.
Verify that every intended table has the tissue column if all images must be
tissue-filtered. Nonfinite areas are excluded when the area filter is active;
the area filter is skipped if its column is absent from all inputs. **Off** skips
that filter explicitly. All original rows remain in the output; excluded rows
have blank corrected values. The optional indicator is 1 for retained corrected
cells and 0 for excluded cells, and cannot overwrite an input measurement.

Post-correction QC measures the Mahalanobis distance of correction vectors in
arcsinh space. A 95th-percentile cutoff identifies roughly the largest 5% of
changes; it does not prove that those changes are wrong. Leave automatic
post-correction exclusions off initially and inspect the affected populations.

## Choosing the arcsinh cofactor

The transformation is `asinh(x / c)`. The cyCombine paper recommends different
cofactors for different cytometry technologies, including 5 for CyTOF. It does
not establish 5 as a CODEX imaging default. Its correction and evaluation rely
on comparing cell populations while accounting for biological conditions.
[Pedersen et al., Nature Communications (2022)](https://www.nature.com/articles/s41467-022-29383-5).

In this repository, Flow gating's **Dynamic** option calculates each marker's
cofactor as a chosen percentile of absolute measurements in the reference cells
(`src/spacemap/flow_transforms.py::fit_step`). The default is the 5th percentile.
It can also fit separately within groups. This is reference-based scaling, not
a direct estimate or subtraction of marker background. Zeros can make the
chosen percentile unusable, and cell-composition changes can alter it.

My recommendation for variable-background imaging data is to establish
background handling first, then compare transformations using predefined
negative and positive reference populations. A marker-specific cofactor recipe
would need to be fitted on suitable reference cells, recorded and frozen across
slides. Re-estimating a percentile independently on each slide could confound
cell composition with transformation scale. Studio correction currently accepts
one shared scalar cofactor; Flow recipes are not automatically applied to it.
Do not pre-transform the input and then transform it again during correction.

Treat 5 as a candidate to evaluate, not a validated optimum. Compare candidate
settings using technical replicates or common cell populations across batches,
positive/negative separation, and biological effects intended to remain.
Similar pooled histograms or improved pooled EMD alone cannot establish a valid
correction. The best cofactor for this project cannot be determined from the
settings screenshot or the magnitude of pooled batch differences alone.

See the [engine review](batch-correction-review-2026-10-09.md) for the Python
cyCombine implementation, recorded R-fixture comparison, and remaining limits.

The [pixel-versus-cell threshold audit](batch-threshold-review-2026-10-24.md)
tests the current threshold methods on project cell means, matched image pixels
and known-truth simulations. It recommends separating cell-positivity gates from
continuous batch-correction inputs; pixel thresholds are not interchangeable
with cell thresholds or arcsinh cofactors.

For inspecting corrected markers before sub-clustering, see the
[Marker review workflow](studio-marker-review.md). Sample grouping is optional
and does not require patient metadata.
