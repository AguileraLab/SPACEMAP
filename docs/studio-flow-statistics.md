# Flow statistics

Open **Stats** beside **Plots** and **Layout**. The population tree, group colors,
image selection, figure folder and workspace are shared with Flow gating.

1. Drop gates into **Comparison groups**, or use **Active Plots → Import from Plot**.
   An import retains each series’ exact parent hierarchy, filters and fitted axis
   transformation. Turn off Live view before importing a reproducible comparison.
2. Choose **Group By** (normally patient), minimum cells, features and a test.
   The **Fields & tests** drawer also supports dragging into those setup boxes.
   Use the color swatches and group labels to prepare publication labels.
3. **Run analysis**, then switch between comparisons, delta heatmaps and the table.
   **Hide setup** gives the figures more room. Editing controls does not start
   expensive calculations automatically.

### Find and add features

Choose a metric, then search the marker checklist (for example `_corr`). **Select
all matches** checks only the matching markers. Checks remain selected when the
search changes; the selected count includes hidden matches. **Add feature** adds
all checked markers, skipping duplicates. **Clear** resets the selection.

For **Cell count** or **Population proportion (%)**, use the population checklist:

- **All cells in each comparison group** measures the whole group.
- **Category: phenotype = Treg** applies that category independently inside every
  comparison group. Categories come from existing categorical split gates; repeated
  categories across branches appear once. Their column and values are saved with
  the feature, independently of the original split gate.
- **Gate: …** intersects each comparison group with the exact selected gate and
  all its parents. A gate limited to timepoint 1 will contribute zero to a disjoint
  timepoint 2 group. Use a **Category** to compare the same phenotype across branches.

For a Treg comparison, add the two parent cohorts as comparison groups, choose
**Population proportion (%)**, check the Treg **Category**, and keep **Denominator =
Each comparison group**. With **Group By = patient**, each dot is that patient's
Treg count divided by their cells in the corresponding cohort, times 100. Keep the
same biological denominator definition in both cohorts. Use a paired test for the
same patients measured twice; otherwise use an independent-group test.

Counts describe sampled abundance and depend on sampled tissue size. Proportions
describe composition, not absolute abundance; another population's expansion can
change the Treg proportion even if its absolute abundance does not change.

## Histogram thresholds

Facet X ticks and the marker/scale title belong below the lowest panel. The shared
X axis retains aligned gridlines. The newest compatible threshold under the plot
parent is active automatically; **Threshold ▾** selects another or hides its readout.

Legend counts include **all cells in the selected, filtered gate**. The percentage
is the number of finite marker values **strictly greater than** the cutoff divided
by that total. Missing measurements remain in the denominator; cells outside the
visible X range are still counted. Hover a legend to see the cutoff and numerator.

Use the threshold menu, or right-click its line, to apply the same cutoff to all
facets, broadcast to open plots with the **same fitted axis**, or save a named global
threshold. Global thresholds are immutable copies of the marker, fitted transform
and cutoff, so a later gate edit does not silently change an existing feature.
Different fitted transformations do not share a coordinate automatically.

## Statistical unit and measurements

- With a biological-unit column, one dot is one eligible unit. Missing or blank
  IDs are excluded and reported. The minimum cell count applies separately within
  each selected group, or to the denominator for a gate proportion.
- Mean and median intensity aggregate finite marker values, in stored units unless
  an imported feature carries a fitted transformation.
- Percent positive uses the saved threshold’s transformed units and the all-cell
  denominator described above. A positive value is never inferred from histogram bins.
- Cell counts sum the chosen target per biological unit. Population proportions
  divide that target by each comparison group, a selected containing gate, all
  analysis cells, or each comparison gate's own parent. A denominator that excludes
  target cells is rejected. Minimum cells is evaluated on the denominator (the
  comparison group for counts), keeping eligible patients with zero target cells.
  A patient with no denominator cells is unavailable, not a zero proportion.
- Pooled mode uses individual cells for inference and is explicitly labelled. Dots
  and violin shapes display at most 1,500 deterministically sampled cells per group;
  tests, summary statistics and box quartiles use every eligible cell. A pooled
  report contains the displayed cell values and exact complete-group summaries.
  Pooled counts and proportions show one descriptive total per group with no
  p-values. Population inference requires biological replicates.
- Clicking a dot sends that biological unit intersected with its gate, imported
  filters and population-feature target to Studio's image viewer. A plotted sample
  is never used as a gate mask.

## Tests and correction

Two groups support Mann–Whitney U, Welch/Student t, paired Wilcoxon and paired t.
Paired tests join exact biological-unit IDs after aggregation and report matched
and unmatched counts. They require at least two matched units. Multi-group
independent comparisons use Kruskal–Wallis plus tie-corrected Dunn tests or one-way
ANOVA plus Tukey HSD. Paired multi-group requests return paired pairwise tests.

All tests are two-sided. Insufficient groups produce an unavailable result with a
reason, not a zero p-value. Identical constant groups return p = 1. Overlapping gates
and shared IDs in unpaired comparisons are reported in **Analysis notes**; choosing
a biological unit alone does not establish independence.

The correction family is all valid omnibus, pairwise, heatmap and optional
between-cohort delta hypotheses in the current run, across all features.
Benjamini–Hochberg produces FDR q-values; Bonferroni produces familywise-adjusted
p-values. Tukey’s already familywise-adjusted p-value is kept in a separate
`tukey_p` column. ANOVA pair nominal p-values use the omnibus residual variance;
they are the values entered into the cross-feature correction.

The comparison figures show up to 20 pairwise annotations per panel to keep the
figure readable; the table and report contain every comparison. Exact p and
adjusted p, adjusted-significance stars, or no brackets can be selected.

## Delta heatmaps

Use **Absolute values**, **After − Before**, or **Log2 fold change**. Each named
before/after pair becomes a heatmap column; features form the rows. Arrange groups,
features and pairs with the arrow controls; **Transpose** swaps the matrix axes.
Branches can represent clinical cohorts, phenotypes, or their intersections.
The **Metric** control selects the feature type displayed in the heatmap. Showing
all types together retains their different units; use row Z-scores for relative
comparisons across columns.

Patient-matched mode computes a change for each complete pair, then its mean or
median. Cohort mode compares group means/medians without matching. Log2 ratios
require positive values; any pseudocount is explicit and exported. Invalid ratios
remain unavailable. Per-tile tests compare the original before/after metric values
using paired Wilcoxon/t or unpaired Mann–Whitney/Welch, according to the design.
The optional comparison of patient deltas across cohorts is a separate test family
in the result table. Absolute tiles are descriptive and have no fabricated tile p.

Z-scoring standardizes each feature across its displayed columns, using population
standard deviation. It changes display only. Tests and exported raw deltas remain
unchanged. Nominal and adjusted significance symbols have separate cutoffs and
legends. The diverging color scale is centered on zero.
Axis labels wrap, and the matrix, colorbar and significance key have separate
reserved space. Narrow panels scroll horizontally instead of compressing the key.
Absolute-value heatmaps omit the significance key because their tiles are descriptive.

## Exports, reproducibility and performance

SVG/PDF and 300/600 DPI PNG exports use the existing Studio export service and
output folder. Heatmap tiles export as vector rectangles. Multi-panel exports keep local SVG
gradients and clip paths unique.
CSV includes tests, group summaries, unit values, heatmap values and method/design
records. Excel places those sections in separate sheets without requiring an
optional spreadsheet package. Reports distinguish full-group summaries from the
values actually used in matched tests (`tested_a_*`, `tested_b_*`). Every saved
report has a JSON manifest containing the data revision, gate definitions and design.

Stats results reuse the bounded Flow cache (16 entries, 64 MiB on disk), with
source revision, gates, transforms and design in its key. **Jobs → Flow cache** clears
these generated results without deleting workspaces or exports. Colors and display
settings update locally. Changed designs cannot export or overlay stale inference. Source-data revisions
are checked again before figure/report export and patient-dot image selection.

A synthetic 2,480,000-cell check with two markers, two groups and 50 matched units
per group took 3.6 seconds for aggregation and inference, excluding source loading;
the returned result was about 30 KB. This is a benchmark, not a latency guarantee.

Implementation is separated into `spacemap.flow_statistics` (scientific engine),
`studio.flow_statistics` (revision checks, cache, reports), and `flow-stats.js` (UI).
New metrics and tests can extend these modules without changing the gate engine.

Methods follow [SciPy’s statistical tests](https://docs.scipy.org/doc/scipy/reference/stats.html),
[Dunn’s tie-corrected comparison](https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html),
and [Statsmodels’ multiple-testing correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
