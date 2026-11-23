# Phenotyping without Batch correction

Use segmentation or cell-tagging CSV/Parquet tables directly. Set the **Tables** folder in Phenotyping; Studio pairs files with images. Correct a pairing in the dropdown beside each image. For individual image tables, choose **No image column — one image per table**. Tissue-piece numbers such as `tissue_roi` are not image names.

The list shows every paired image's cell count. Opening an image activates its table. Add trusted manual labels to the same column in each image, for example `phenotype_ref1`. Click/box labelling saves quietly, without rebuilding the image list. **Write into table** commits saved edits to the original CSV or Parquet; a CSV's Parquet cache is only a reading aid.

## Load, label, train

1. Set **Models** at the top to a trained-model directory or shared library; press **Use**.
2. Under **Model training**, select a model and press **Use**. That model supplies suggestions and is the starting point for updates. Only load model files from people you trust.
3. Accept a suggestion or type your own phenotype. A prediction does not become a reference label until you assign it.
4. Keep or adjust the **Split**, for example 70% training / 30% validation.
5. Press **Train model** to learn from this image, or expand **Learn from several images**, select your images and press **Train selected**.
6. Alternatively, turn on **Auto-train this image** beside the training button. After a pause in labelling it trains from this image using the currently active model. It updates at most once a minute, not on each click.

Both manual and automatic training save dated versions in the Models folder and activate the new version for suggestions. Previous versions remain available. The active model follows you when you switch images in this project; enable automatic updates separately for each image if desired. In another project, select the same Models folder and use the version you want. The model does not train until you ask it to, or enable automatic updates.

A new phenotype belongs in the same manual column, such as `phenotype_ref1`. Accept a suggested label or enter a new name. Give a new type at least five usable examples. Updating can produce a model with both old and new phenotypes when the loaded package contains reusable training examples. Older Studio models are supported when their saved examples identify the rows used in the final fit. A classifier without examples can still predict; turn off **Keep previous training examples** to train from new labels only.

**Details & options** contains the storage location, optional version name, alternative training-label column, and **Keep previous training examples**. Normal use needs no changes there. The visible training summary names the column and whether old examples are included. A fresh model produces Random Forest and Extra Trees versions; a loaded model retains its classifier type. The dropdown identifies each version by method, date and ID.

**Keep previous training examples** is enabled when you use a saved model:

- **On:** refit a new classifier from retained training examples plus this project's training labels, using shared feature columns. Older examples from the current input tables are replaced, so repeat training does not duplicate them or reuse current validation cells.
- **Off:** train only on this project's labels and selected features.

Measurements missing from older projects are never invented or filled with zero. A combined fit therefore cannot use a new marker absent from an older retained panel. Full new-panel training examples remain in the package, even when some measurements are unused by the combined classifier. Use current-project training to fit all newly selected markers. Additional data does not guarantee improved performance; feature names, units, preprocessing and phenotype definitions must agree across projects.

Training requires at least five usable new labels per phenotype. Smaller groups are listed as skipped; other usable phenotypes can still train. Previously learned classes remain available when their saved examples are retained. A fresh model needs two phenotypes; with saved examples, a single new phenotype can be added if the combined training set has at least two types. Sampling is bounded: up to 2,000 cells per phenotype and 50,000 total (10,000 for Auto-train). The replay archive retains at most 50,000 training examples across projects; older examples are reproducibly sampled when needed. Validation examples are stored separately and never replayed into training. Old Studio versions fitted all saved examples after an internal check. Those identifiable final-fit examples can be reused as training data; their old internal check is not treated as independent validation. New versions retain validation examples separately across projects, replacing a table’s old validation partition when that table is trained again.

**Training labels**, under **Details & options**, chooses the target categories. Normally it follows your selected manual column, such as `phenotype_ref1`. Other phenotype columns are allowed. `bed` appears under **Other table labels** because it describes annotation regions, not cell types. Training on another method's assignments learns those assignments; it does not independently validate them.

Leave **Use checked labels only** off if all your manual assignments are trusted. Turn it on only when using **Mark checked** as an additional review step. Cells marked Needs review are excluded either way.

## Predict in MR-ML

Choose **MR-ML → Use a saved reference model**. The loaded model is offered automatically. Choose the active table or **Selected image tables**; no Batch correction is required.

Open **Phenotypes to predict** to restrict the output classes. The available-feature check evaluates saved validation examples from the retained projects using the target table's feature columns. It shows how many learned phenotypes are suggested and marks the others **needs checking**. There is no eleven-phenotype limit. A missing marker changes this recommendation, not the saved phenotype list. A matching panel permits all learned classes to be scored; it does not guarantee accuracy in a new tissue. **Select supported phenotypes** selects classes with at least three saved validation examples and precision and recall of at least 0.50. These are screening criteria, not evidence of accuracy in a new tissue. Missing markers can make classes indistinguishable; review them in this project.

An unselected winning phenotype stays unassigned. The model does not force such cells into a selected class or renormalize their scores. **Keep existing reference labels** optionally copies manual labels into the result; turn it off when preparing columns for F1 assessment. Prediction never changes `phenotype_ref1` itself.

Manual-reference suggestions use the same available-marker check. A **check** label on a suggestion means low validation agreement or too few validation examples for that marker set. It does not delete the phenotype or prevent you assigning it. Suggestions for several selected cells carry this flag if any cell's available panel needs checking.

Output names and confidence filtering are configured here, not under Manual reference. Each image keeps its own prediction columns. Results are initially saved as sidecar columns, then committed to CSV/Parquet with **Write into table**. Completed images retain their results if a later image fails. Existing columns require an explicit overwrite choice.

## Assess methods and export figures

Use **Assess methods (F1)…**, or **Visualization → Method agreement**.

- Choose a reference column and one or more prediction columns.
- Choose **Validation cells**, **Training cells**, **Training + validation**, or **All labelled cells**. For the first three, select a saved training split.
- Press **Compare**. Scores and Classic F1 plots appear in the Figures panel beneath the image.

A split records the exact cell identities and labels. It is reused across methods without reshuffling, survives model retirement and adding result columns, and refuses reordered cells or changed reference labels. Missing predictions count as unassigned; methods are compared on the same cells.

Available figures: micro/macro/weighted F1 comparison, per-phenotype F1, precision/recall/F1 heatmaps, confusion matrices and F1 by image. Export PNG, SVG or PDF, or download scores and the exact evaluated cell list as a ZIP of CSV files.

Training cells can assess a method that never used those labels. For a fitted reference model, use held-out validation cells and actual predictions, not copied manual labels. Repeated model selection or label editing after viewing predictions can bias validation. A cell-level split is not an independent-slide test.

## Keep and share

To split broad phenotypes by region and functional markers, see [Sub-phenotyping in Studio](subphenotyping.md). Saved models and manual labels can help name the subsets.

New manual and automatic versions are saved under:

```
<Models folder>/phenotyping/phenotyping_<version ID>/
    model.json
    <version ID>.joblib
```

Keep the whole version folder. Another user selects the library in **Models**, chooses the version and presses **Use**. **Details & options → Stored at** shows its location. **Save a copy in Models folder** is needed only for a model stored elsewhere or a failed export. If the destination is full or read-only, Studio reports the problem and keeps the local version for recovery. Automatic updates also create separate versions, so their storage use grows over time.

Packages include retained training/validation measurements and labels, but no source images or full cell tables. Consider those sampled measurements when sharing. **Project → Model library** also handles Segmentation (Cellpose) and Artifact models; these model types are separate from phenotyping classifiers.

## Shared project writes

New project/output directories inherit the parent directory's group. Label files, review databases, pending columns and atomic JSON updates receive group read/write permissions. Collaborators must belong to that group. SPACEMAP does not hard-code a lab group or grant access to unrelated users.

Your account cannot repair files owned by another account. Their owner can use `spacemap share-project /path/to/project`. A selected-cell counter in the viewer with “Select cells first” in the label panel indicates a selection-link issue, not necessarily a permission error.
