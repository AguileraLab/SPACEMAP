# Phenotyping models

## Manual reference

1. Set **Models** to the folder containing your models. Click **Use** beside the folder.
2. Under **Model training**, select a model and click **Use**. Its suggestions help you label cells; you choose the final labels in `phenotype_ref1`.
3. Under **Train on**, click **Current image** to learn from the open image. Or tick images in the table above and click **Selected images** to learn from those together. Both individual CSV files and merged tables work.
4. Optional: enable **Auto-train this image** to train after you pause labelling. A new phenotype needs five usable labels. Previous training examples are kept by default; change this under **Details & options**.
5. Each successful training saves a dated version in **Models**, selects it, and keeps the original. **Details & options → Model folder → Copy path** locates its export. Copy the whole model folder to share it.

The default split is 70% training and 30% validation. Validation examples stay separate from training. Batch correction is optional. If no image is open, use **Selected images**. Progress and errors appear below the training buttons; explanations are in the **?** boxes.

Only the selected Models folder supplies the model list. An empty folder lists no saved models. Newly exported folders have readable names and the browser's time zone. Existing exports keep their original names.

## Models review and edit

1. Select one or more models and click **Inspect models**. Review their phenotypes, shared features, unique features and saved examples.
2. Inspect **Figures** on the right:
   - **Reference feature profiles:** mean feature measurements for each labelled phenotype, standardized within each model. These show example patterns, not a decision rule.
   - **Feature importance:** overall use of each feature in the trees. This is not a per-cell explanation.
3. Click **Validate models** to compare them on the same saved validation cells. The confusion matrices show reference labels in rows and predictions in columns. Cells used for training by any selected model are excluded. If shared validation is unavailable, use **Compare predictions (F1)** with labelled cells in your project.
4. Optional edits:
   - **Features to keep → Retrain selected model:** exclude an input and fit a new version. Fix incorrect measurements in their source table before training again.
   - **Group phenotypes → Save custom copy:** give Normal, Tumor and Sarcomatoid the name Epithelia. Detailed types stay in the model.
   - **Merge selected models:** fit a new Random Forest from their saved examples. Repeated cells count once. Conflicting labels or measurements must be resolved first.
5. Enter an optional **New model name**. The new version saves in Models; select it under Model training and click **Use**. Originals stay unchanged.

Merging requires reusable, identifiable examples and shared measured features. The new fit uses the selected shared features; unique features remain archived for later use. Models without examples can still predict and be copied with grouped output names. A merge is currently limited to 12 models and 50,000 unique training examples.

## Prediction and assessment

Use **MR-ML → Use a saved reference model** to predict the images ticked above. You can filter cells, limit output phenotypes, and choose a new output column. **Use groups saved in the model** applies broad names; **Save detailed labels too** adds a column ending in `_detail`. Turn off saved groups to predict detailed types instead.

**Compare predictions (F1)** opens Visualization. Choose your prediction columns and manual-reference column. Use validation cells for models trained from those labels; training or all labelled cells can assess independent methods. Higher F1 means better agreement with your labels, not proof of accuracy in another tissue.

Models trained with different panels use shared measurements. Missing features do not by themselves prove that a phenotype is predictable; check the available-feature validation and the new tissue.
