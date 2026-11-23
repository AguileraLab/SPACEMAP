# Phenotypes used by reference models

In **Phenotyping → Manual reference → Model training → Phenotypes · Train / Predict**,
each phenotype has two independent switches. The same control is available in
MR-ML. Search narrows the list; each header checkbox changes only matching names.
Switches save immediately for the current project.

| Switch | When off |
| --- | --- |
| Train | Excludes that label from future training, including replayed examples and saved-example refits/merges. |
| Predict | Prevents saved-model predictions and cell-card suggestions from choosing that label. |

To use **Others** only for uncertain cells:

1. Untick **Train** and **Predict** beside **Others**.
2. Train a new model version to remove Others from its learned classes, or apply an existing model with Predict off.
3. In **MR-ML → Scores and output columns**, keep **Below cutoff = Others** and choose **Minimum score**.
4. Run prediction again. Existing result columns are not changed by a switch alone.

For example, an old model gives Others 80%, CD4 15%, and CD8 5%. With Predict off
for Others, the base prediction becomes CD4 with **15%** support. It does not
become 75% by redistributing the blocked class's probability. With a 0.75 cutoff,
the Refined result becomes Others. These scores describe model support, not
calibrated certainty. Review performance after changing the allowed classes.

Manual annotations and original model files stay intact. Training writes a new
version. Turning Predict back on makes the class available in an old model;
a newly trained model that never learned it needs new training to add it back.
At least two usable classes are required to train, and at least one allowed class
to predict. Low-confidence fallback labels are independent of trained classes.

The optional **Validation and output subset** control retains its previous meaning:
an otherwise enabled class outside that subset can win, leaving the output
unassigned. The explicit Predict switches remove a class from competition.

Existing Classic MR-ML label exclusions remain effective until the user saves
these shared switches. Once configured, the switches replace that older list.
Queued jobs keep their saved policy; the new choices apply to subsequent runs.
Model validation reports describe the saved model unless predictions are explicitly
rerun and assessed with the new policy. No validation score is silently recomputed.
