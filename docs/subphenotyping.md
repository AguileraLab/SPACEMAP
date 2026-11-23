# Sub-phenotyping in Studio

Use **Phenotyping → Sub-phenotyping** to split an existing cell type into smaller groups. Check the images to process in the table above. To process one ROI, check only that image. Multiple images must share a merged cell table; a single image can use its paired table. Batch correction is optional.

1. **Choose cells.** Select the broad phenotype column and a type, such as **Epithelia**. Add a region filter such as **tumor is 1**. The selected-cell count updates before you run.
2. **Find subsets.** Check your markers, enter a new **Save as** column, such as `phenotype_tumor_subsets`, and click **Run sub-phenotyping**. Run uses every checked marker. **Find markers** is optional: review its suggestions, then choose **Use suggested** or edit the per-phenotype checkboxes if you want fewer markers. **Use all selected** restores the full panel.
3. **Name and save.** Review the marker heatmap in **Figures** and the clusters on the image. Type names such as **Tumor**, **Sarcomatoid**, or **Proliferative Tumor**. Give clusters the same name to merge them. Click **Save names**.

Under **Help name subsets**, you can choose a saved model from the Models folder and a manual-label column such as `phenotype_ref1`. **Limit model suggestions** lets you consider selected types. If an excluded type wins, the panel says so; it does not force a different label.

Model scores and manual-label counts are shown separately. Scores summarize up to 400 sampled cells per cluster; they are not accuracy measurements. **Check** means limited validation, low agreement, or missing measurements. You decide the final name.

Region filters use columns already saved in the cell table. To use a GeoJSON region, first apply it through Annotation/cell tagging. Multiple filters must all match. Clustering needs at least 100 selected cells and two kept markers per parent phenotype.

Switching between raw and corrected marker values keeps the checked marker names when both versions exist and resets per-phenotype subsets to the full checked panel. Changing preprocessing or images preserves explicit marker subsets but clears outdated suggestions. The heatmap shows the markers actually used by that run; changing settings does not update an earlier result.

Results stay in new columns. Unselected cells remain blank in those columns; the original phenotypes and manual labels remain unchanged. Studio saves result columns as Parquet sidecars. Use **Write into the CSV** in Jobs when you want them in the CSV itself.

## Preprocessing and heatmap contrast

Open **Find subsets → Preprocessing**:

| Setting | Suggested starting choice |
| --- | --- |
| Transform | **Arcsinh raw; keep corrected**. Applies `arcsinh(value / cofactor)` once to raw markers. SPACEMAP `_corr` values are already in arcsinh space. |
| Cofactor | Start with the acquisition/project's established value. The default is 5, but no single cofactor suits every imaging assay. Use the same value across slides. |
| Normalize | **Z-score each marker** gives markers comparable weight, including weak/noisy markers: review the panel first. **None** retains the transformed scale but lets high-variance markers contribute more. **Median / IQR** is an alternative; zero-IQR markers fall back to SD to retain rare signal. |
| Within | **All selected cells** by default. Select a metadata column such as **Slide** to normalize each slide separately. All selected ROIs with the same slide value are pooled. |

Normalization is fitted separately within each selected parent phenotype, after image and region filtering. Normalizing within Slide removes slide differences, which may include real biology. It is not a substitute for validating batch correction. For already corrected data, start with **All selected cells**.

Missing measurements are imputed at the normalization group's center; normalized values are clipped at ±5. **None** keeps the transformed measurement scale, imputes missing values with the selected population's median and does not clip. The job saves transformation settings and fitted centers/scales in `preprocessing.json`; source measurements are unchanged.

The right-hand cluster heatmap standardizes each marker's cluster summaries again for display. Red means higher **relative to other displayed clusters**; blue means lower. Pale colours do not prove the transformation is wrong. **Colour range → Auto** fits the displayed values. A smaller fixed range, such as **±1.5**, adds contrast but saturates extremes. The legend and exported figure use the same range. This changes neither measurements nor cluster membership. Use the same fixed range when comparing figures from different runs.

Annotation's pixel classifier uses arcsinh (optionally a per-marker background-based cofactor), percentile scaling, spatial smoothing and Z-scores. Cell means have different noise distributions, so its smoothing and pixel-derived cofactors are not copied into cell subclustering.

For biological positive/negative calls, inspect marker distributions and image controls, then set marker-specific gates. Clustering Z-scores and attractive heatmap contrast cannot define positivity. For comparison, [Scimap's rescaling documentation](https://scimap.xyz/Functions/pp/rescale/) distinguishes manually defined gates from GMM estimates and notes that per-image GMM needs enough variation to distinguish populations. Automatic gating is not added by these preprocessing controls.

### Exploring proliferation within epithelial cells

For this question, first confirm epithelial identity and segmentation on the image.
Use Ki67 to assess proliferation-associated expression; review nuclear signal and
weak/borderline examples across slides before setting a gate. A heatmap-high group
is not automatically Ki67-positive, and Ki67-negative does not prove permanent
quiescence. [Miller et al.](https://pubmed.ncbi.nlm.nih.gov/30067968/) describe Ki67
as a graded marker rather than a simple binary measure of proliferation.

For exploratory clustering, choose markers that address the intended state and
have convincing signal above background. Epithelial identity markers can remain
useful for checking the result without necessarily driving a proliferation split.
An uninformative marker should not receive equal weight merely because it can be
Z-scored. Conversely, a weak but clearly specific signal should not be discarded
just because its absolute intensity is lower than other markers.

Screening functional markers across phenotypes is a useful first step, provided
the screening uses measured intensities and image review rather than only
relative heatmap colours. Then check variation inside the phenotype to split:
reliable positive/negative populations or a meaningful intensity gradient can
support clustering. Mere presence is insufficient when almost every cell is
uniformly positive or differences mostly reflect background. Within-phenotype
Z-scoring is reasonable for such a reviewed panel, but its high values remain
relative expression, not positive calls. Review the resulting groups across
images and slides before assigning biological names.

With corrected markers, **Arcsinh raw; keep corrected** and **Normalize: None**
provide a useful baseline that adds no new variance scaling. This is not a claim
that unscaled clustering is universally best: compare reviewed groups with a
Z-scored run of the same reliable marker panel. Multiplexed-imaging benchmarking
found Z-scores useful for some weak, specific markers, while fine cell-state
annotations were less reliable than broad cell-type labels
([Hickey et al.](https://doi.org/10.3389/fimmu.2021.727626)).

The current **All selected cells** option fits normalization inside each selected
parent phenotype. It does not reuse a scaler fitted to all cell types before
filtering. A fixed whole-dataset reference could change relative marker weights,
but would still not establish a positive/negative threshold. Review actual
measurements and image signal, plus consistency across slides; relative heatmap
colour alone cannot distinguish biology from background or segmentation spillover.

## Review subsets on the image

Click a heatmap row or a cluster in **Name and save** to show its cells; Ctrl/Shift selects several. The **Data** tab automatically filters to those cells within its current image/dataset scope. Toggling groups in the cell legend also updates Data. **Show all** clears the cluster filter. The image, cell legend, heatmap row swatches and naming fields share the same colours, including draft edits. **Save names** writes those names; until then they are a preview.

`phenotype_sub_clusters` keeps individual cluster IDs; `phenotype_sub` holds saved phenotype names. Several clusters may share a name and colour while remaining individually selectable. Both use one cell overlay. **Show on the image** reuses it, and old duplicate result overlays are removed when the image reopens. Hidden clusters lose both their fills and outlines.

## Explore markers before clustering

Open **Marker figures** under the existing **Marker panel**.

1. Check the images, then choose the phenotype column, phenotypes, region filters and markers above. Untick **Others** to exclude it.
2. Under **Figures to generate**, tick the views you want. Only **Dot plot** starts selected; your choices are remembered.
3. Set the options shown for those views, choose **Save in**, then click **Generate selected**. **Patient heatmap** needs a patient column; **Group comparison** needs **Stratify by** (such as arm + timepoint + response). Co-expression needs two markers.
4. In the lower-right **Figures** pane, choose a figure. **Show markers** displays one or several markers; **Expand** opens a large view. **Show saved figures** reopens the latest finished run. Older runs remain in **Jobs → Show figures**.

In the right figure pane, open **Appearance** to adjust text size, X/Y label size and angle, width, height, wrapping and label spacing. Heatmaps also have **Show values**. Click **Apply view** to update the preview and its **PNG/PDF** downloads. Automatic layout gives long patient/group labels more room. Each figure remembers its applied settings in this browser; **Reset appearance**, then **Apply view**, restores the defaults. **Run figure book** contains the original run's figures.

| Figure | Read it as |
| --- | --- |
| Marker × phenotype | Dot area: percentage high **within the phenotype**. Colour: median expression, standardized per marker. |
| Marker distributions | All measured cells; median, middle 50%, high cutoff and equal-patient outline. |
| High expression by patient | Percentage high within each patient's phenotype. Grey means too few measured cells. |
| Patients by group | Each patient and combination of group values is separate, including repeated timepoints. |
| Group comparison | Mean of eligible patient percentages, with patient counts (`n`). Without a patient column, percentages use pooled cells and are labelled accordingly. |
| Phenotypes among high cells | Which phenotypes make up the **high cells** for each marker; these bars sum to 100%. |
| Marker variation | Markers ranked from most to least variable within each phenotype. Default: middle 50% spread (IQR). |
| Marker combinations | All measured cells for the selected pair, binned by expression; colours show cell counts. |

Every marker uses one cutoff: **mean + Z × SD across the selected cells**. It stays fixed across patients and groups. Changing the cell selection and generating again changes the reference population. “High” is exploratory, not a validated positive gate.

Figures share the **Preprocessing → Transform** and **Cofactor** settings. Choose **Use stored values** for other already-transformed columns. Figures use these transformed measurements before clustering normalization, so patient/slide differences remain visible. Missing measurements are excluded per marker, never filled with zero. Missing patient/group IDs are excluded from the corresponding grouped summaries, with counts recorded in the metadata.

Variation is descriptive: staining differences and noise can also produce a large spread. Optional relative IQR divides the within-phenotype IQR by that marker's SD across the full selection. A high score alone does not establish a useful biological subtype. Patient comparisons are descriptive, not significance tests.

Each run generates only the selected figure types, in a new folder containing individual PNG/PDF figures, `figure_book.pdf`, Parquet measurement summaries, and JSON settings/provenance. Unselected distribution and patient calculations are skipped. Filenames include the phenotype where relevant. The source CSV/Parquet is unchanged. Display-only marker choices use the saved summaries and retain the original cutoffs. These figures use the shared **Marker panel**, not the later per-phenotype “kept for clustering” lists.

## Broad labels first, finer types later

In **MR-ML → Use a saved reference model → Group output names**, give **Normal**, **Tumor**, and **Sarcomatoid** the same output name: **Epithelia**. Their model probabilities are added before choosing the output. The model retains its original types.

Then use **Sub-phenotyping** to select **Epithelia** in **tumor is 1**, choose the same model, and limit suggestions to **Tumor** and **Sarcomatoid**. Choose functional markers to explore further subsets. Save a separate column for each analysis, such as `phenotype_tumor_subsets` or `phenotype_stroma_subsets`.
