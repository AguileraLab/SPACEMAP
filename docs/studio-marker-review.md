# Marker review

In **Phenotyping → Sub-phenotyping → Marker figures**, select **Marker review**
and click **Generate selected**. This replaces Studio's separate Find markers
button. Existing individual figures remain available.

1. Choose the images, phenotype(s), region filters and markers above the figures.
   The review uses exactly this selection. Include other phenotypes if you want
   their distributions as context; selecting only one phenotype is also valid.
2. Use validated batch-corrected markers for comparisons across slides, and
   inspect original measurements and staining before interpreting a marker.
   **Arcsinh raw; keep corrected** transforms raw columns once using the selected
   cofactor and leaves `_corr` columns unchanged. Choose **Use stored values**
   for custom columns already transformed. Review measurements precede clustering
   normalization: a review IQR is not a Z-score unless the stored inputs were Z-scores.
3. In **Figure options → Compare samples by**, optionally choose a sample column.
   It can be subject, specimen, acquisition batch or any imported category. **None**
   works without metadata, including a single image. Missing sample IDs stay in
   pooled measurements but do not contribute to sample comparisons. Small samples
   are flagged. Choose the unit that answers your question; differences between
   subjects are not automatically technical batch effects.
4. Open a marker in the review. Compare its histogram, middle 50% spread (IQR),
   5–95% spread, measured-cell percentage, phenotype context and optional sample
   medians. A star suggests inspecting a marker; it does not establish positivity.
   Click a histogram bar or enter a range, then **Show cells**. Data and the current
   image use the same cells; the image and zoom stay in place. Data initially shows
   this image; its Whole dataset control can show the complete selected range.
   **Clear selection** or **Show all cells** ends this temporary filter.
5. Confirm the staining and segmentation on representative cells and samples.
   Exclude suspected background/spillover. Tick the markers you decide to use,
   then **Use selected for clustering**. This saves a per-phenotype subset of the
   existing marker panel. It does not run clustering or assign positive labels.
6. Run **sub-phenotyping** and inspect the resulting groups on the original image.
   Within-phenotype Z-scoring weights every variable marker, including weak noise;
   choosing credible signals first matters. A high Z-score means relatively high
   within the fitted population, not necessarily biologically positive.

The shortlist is descriptive: among markers with enough finite measurements and
nonzero variation, average their ranks by IQR/reference SD and 5–95% spread/reference
SD; suggest up to six for inspection. Reference SD uses all selected cells. With
only one phenotype this mainly compares distribution shapes. The rank is not a
validated biological score; rare states and technical effects can defeat it.
There is no universal IQR or positivity cutoff. Independent references are needed
to measure actual positivity accuracy.

Exports include a combined PNG/PDF overview of the first six markers in inspection
order, a figure book, full Parquet summaries and a versioned review record with
selection, transformation and source revision. Every selected marker remains
inspectable in the interactive review. No measurements or labels are rewritten.
Changed source data blocks live selection; changed input settings block applying
old choices. Generate a new review when either changes. Image linking requires
correct image and segmentation-label roles; figure generation does not.

Scientific context: imaging normalization should be checked against staining and
known biology, rather than selecting a method from spread alone. See
[Hickey et al. (2021)](https://doi.org/10.3389/fimmu.2021.727626). The inspection
rank above is a transparent exploratory heuristic implemented in Studio, not a
validated method from that paper.
