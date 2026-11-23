# Gland types and exterior niches

Open **Spatial analysis → Gland types**. Classic Tab 8 has the same method and
uses the same engine, settings, jobs and result files.

## Use it

1. Check **Images**, **Tables**, and **Output**, then tick the images to fit
   together. These sources are shared with Phenotyping; separate CSV/Parquet
   files do not need to be merged.
2. In **Annotation columns**, give each compartment a name. Choose its gland
   ID column, matching exterior bed ID column, and optional bed flag/value.
   Choose one or several compartments in **Fit scope**.
3. Choose the **Interior** subtype column and its labels. Choose the **Exterior**
   cell-type column and its labels. Fibroblasts, endothelial cells, and any other
   classes can be included or excluded. The selection defines each denominator.
4. Check **Cell roles and comparison columns**. Image and tissue-piece IDs
   distinguish glands with reused local numbers. X/Y must be image pixel
   coordinates for viewer linking; segmentation labels must match the image's
   cell outlines. Optional **Parent cells** restricts the interior counts.
5. Start with prior **10**, fit minima **20** interior cells and **10** exterior
   cells. Small nonempty structures are kept as predictions, marked
   **Small: review**. Empty or unmatched beds remain unassigned. **Review optimal
   K** tests K=2…8 without adding columns. **Gland types (K)** means the number
   of interior composition types; **Niche types (K)** means the number of
   exterior cell-mix types. Their `?` buttons include examples.
   Review opens two bar charts in the right dock, with the highest separation
   score highlighted. It searches independently of your manual K values: a
   review can suggest eight niches even if the manual setting is four. **Use
   K=…** copies the suggestion into that setting and turns off its Auto checkbox;
   press **Fit** to apply it. Alternatively, **Auto gland K** / **Auto niche K**
   searches during the actual fit. With Auto off, Fit uses your manual K.
   The existing saved reviews also show these charts without rerunning. A
   silhouette suggestion is not proof of the true number of biological types;
   a winner at the largest tested K is marked explicitly. **Profiles** keeps the
   composition heatmaps separate from K-selection charts.
   **Individual silhouette scores** appears beside each K chart (below it in
   a narrow dock). **Fit quality** also opens these charts for a fixed-K run.
   Each dot is one scored gland or bed; the thick line spans the middle 50%
   and its dark tick marks the median. Each row reports its scored count and
   the percentage below zero: these structures resemble another type more
   closely on average. A singleton scores zero by convention. Scores use the
   same reproducible sample of at most 2,000 fit structures as the K chart;
   small predicted structures are excluded, and unsampled types are identified.
   This describes the saved model, not a newly typed K, and is not accuracy or
   independent biological validation. PNG and individual-score CSV exports
   are available. Existing runs are supported from saved counts and models,
   without refitting, changing labels or reading the full source cell table.
6. Press **Fit: Selected**. Progress appears beside the action and in Jobs.
   The current image stays open. **Profiles** shows relative enrichment as
   colour and measured proportions as percentages; figures open in the right
   dock. Hover a percentage to see its Z-score. Click a type name to colour all
   its assigned cells in the **currently open image**, link the same selection
   to Data, and zoom to a local example nearest the fitted centre. The **Cells
   and maps → Colour by** control follows the selected gland/niche column.
   If the type is absent, a message appears below the figure and the image stays
   unchanged. The map legend uses the same action. **Show all cells** clears the
   selection. K-review profiles first require **Fit** to create cell assignments.
7. **Explore** shows both spatial maps and a sortable gland table. Click a
   coloured point, **Interior**, or **Bed** to open the right image, zoom to the
   structure and select its exact cells in the viewer and Data. Display maps
   sample at most 25,000 cells; fitting and linked selections use all cells.
8. For **Compare**, configure the independent sample, visit, group, and optional
   stratification columns before fitting. Choose Group A/B and Before/After
   values explicitly. Review a visit, pooled visits, or the after-minus-before
   change. The table states **Higher in [group]**, group sizes, means, medians,
   p and q. Figures show one dot per independent sample and composition bars
   for each sample/visit.

## What the numbers mean

A **gland** is one annotated epithelial structure. A **gland type** groups
structures with similar interior cell mixtures. For example, **Gland 4 ·
epithelia_pdl1 (64%)** means type number 4 has about 64 of every 100 counted
interior cells in your `epithelia_pdl1` subtype, on average. Individual glands
can differ. It does not mean 64% of all glands, or a genetic clone.

A **niche** here describes the surrounding bed's cell mixture. **Niche 2 ·
Fibroblast + Endothelial** means type number 2 stands out for these two cell
types relative to the other niches. It can contain other cells too; the name
does not claim the two cell types physically touch or interact. As an
illustration, a bed could contain 50 fibroblasts, 10 endothelial cells and
40 other cells and match this type. Those example numbers are not a measured
profile or a required cutoff. K counts these **types**, not the actual glands
or beds found in the tissue.

The raw composition is the selected subtype count divided by all selected
subtype counts in that structure. The smoothed composition is
`(counts + prior_strength × pooled_proportions) / (selected_cells + prior_strength)`.
The prior is estimated from the selected cells across the fitted compartments
and images; it is not estimated separately for each clinical group.

Only structures meeting the fitting minimum determine the weighted means,
standard deviations and K-means centres. Their weight is `log1p(selected_cells)`.
With standardization enabled, the smoothed features are Z-scored using those
weighted moments and clipped to ±3, as in the reference prototype. Constant
or unobserved features in the measured fit compositions contribute zero, so the prior alone cannot create a size-dependent signal. With it disabled, K-means uses smoothed proportions.
The same fitted transform predicts smaller nonempty structures.

Names identify the strongest relative enrichment (or the largest proportions
when standardization is off). Percentages in the profiles are **measured,
unsmoothed** proportions, averaged across fit structures with log-size weights.
Raw counts, fitting status and distance to the fitted centre accompany each
gland. Relative Z does not establish marker positivity or genetic clonality.

K-means uses 20 starts and the saved seed. Auto evaluates K=2…8, constrained by
the number of distinct fit structures, on the same reproducible silhouette
sample of at most 2,000 structures. The method follows scikit-learn's
[weighted K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
and [silhouette definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
The exact transform, priors, centres, names, feature order and library version
are saved as JSON; no model needs to be retrained merely to reopen results.

Clinical burden sums `log1p(selected_cells)` by type and divides by the total
weight within each independent sample and visit. Interior and exterior use
their own cell counts. Thus, a 1,000-cell gland weighs about 6.91 and a 10-cell
gland about 2.40. A missing compartment/bed is missing; an absent type in an
otherwise measured compartment has zero burden.

Pooled comparisons average visits **within** each sample first. Changes require
both visits from that same sample. Two-sided Mann–Whitney tests compare
independent samples, including comparisons of their paired changes; glands
and cores are not counted as independent patients. Groups and strata must be
constant within a sample for this design. By default at least three samples
per group are required. BH FDR includes all planned types, visits, pooled
comparisons, changes and requested strata in the displayed comparison plan;
untestable contrasts count conservatively as p=1 in that family. Statistics are
exploratory, not independent validation of the fitted types.

## Saved data

- `gland_clone` describes the gland containing an interior cell;
  `gland_bed_niche` describes an assigned exterior bed cell. Both names are editable.
  Other phenotyping columns are preserved. Overwrite requires an explicit choice.
- Studio writes these columns to its existing linked result sidecars. The source
  CSV/Parquet bytes are unchanged. The columns are available to Cell data, Data,
  and the existing Jobs write/export controls. Classic also keeps models and the
  augmented runtime dataframe in session state.
- Every run exports under **Output/gland_types/studio_[run ID]**. This includes
  gland counts/metadata/assignments (CSV and Parquet), raw and Z profiles, profile
  PNGs, fitted model JSON, settings, sample burdens and comparison statistics.
  Job originals remain available if copying to the chosen output directory fails.
- Images and tissue pieces never share gland identities. Conflicting interior
  or exterior assignments are rejected, rather than silently choosing a gland.
  Source changes during publication prevent a stale write; a failed multi-column
  or multi-table write restores the earlier result columns.
