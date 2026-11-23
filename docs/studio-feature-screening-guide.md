# Spatial feature screening in Studio

Use **Spatial analysis → Feature screening** to measure spatial features and
compare **independent samples**, across one or several predefined distance zones.
The source CSV/Parquet and its annotations are read-only.

## Separate image tables

Batch correction and a merged table are optional. At the top of **Spatial
analysis**, use **Images** for your chosen images and **Tables** for their
CSV/Parquet files. These folders and table assignments are shared with
**Phenotyping**. TIFF and QPTIFF images are supported; H&E registration is optional.
Changing tabs or these folders keeps the current viewer image open. Click an image
in the list when you want to switch. **Output** is separate and holds Spatial results.

For a file containing one image, use **No image column** in **Column roles**
and select its unique Cell ID (often `label`). Check the **Cell table** dropdown
beside each image. Filename matching ignores known pipeline suffixes and minor
spelling differences while preserving numeric sample identifiers. Missing or
ambiguous assignments stay visible as **No paired table**; choose the right
file directly in that row. Clicking an image opens its exact file from **Images**,
not a similarly named raw acquisition elsewhere in the project.

Choose a phenotype column present in every selected file, then use **This image**
or **Selected**. Cell labels may repeat between images. Spatial searches always
keep different images' coordinate spaces separate, even when tissue IDs repeat.
For Feature screening, `batch_id` identifies the image assignments in memory;
the original files do not need that column. Choose the actual independent
biological sample column for statistical comparisons, rather than assuming
each image is an independent sample.

No merged measurement CSV/Parquet is created. Neighbourhood and TLS results save
with their own source tables using Studio's result journals, and load with those
tables in the viewer and Data. Jobs retains the usual **Write into the table**
action. Selecting one image leaves other image tables unchanged. Counts load
separately so large CSVs do not delay the selectable image list.

Completed runs export under **Output**, in `neighbourhoods`, `tls_candidates`,
`cell_interactions`, `density`, `triads`, or `feature_screening`, according to the
method. Each run has its own folder. Jobs keeps its existing progress and review
records. Changing Output affects new runs, and does not move earlier results.

## Quick start

1. **Tick images** in the list above Feature screening.
2. **Cells and regions:** choose the cell-type column and types you want to study.
   Add a region filter if needed. In **Where to measure**, choose a saved distance,
   annotation polygons, or the whole filtered region. Enter distances in µm.
3. **What to measure:** tick one feature or All seven. Open **Settings** beside
   a feature only when you need to change it. **Diversity** offers four independent
   checkboxes and **All four**. Each selected measure gets its own findings and
   exported figure, with its own scale. All selected measures enter the same
   declared correction groups. Existing single-measure settings are retained.
4. **What to compare:** choose **Independent sample ID**. Three images from one
   person must share one ID and count as one person. Choose the comparison column
   and Group A / Group B. For one timepoint, filter to it in Cells and regions. For before/after
   in the same subjects, use **Match groups within each sample**.
5. **Screen: Selected.** Results appear during the scan. Final figures show the
   sample counts, effect, p and q. Click a dot to inspect its cells in the image
   and Data. The result header identifies the saved run's sample ID; changing
   the controls does not recalculate that run.

**Density** means cells per mm². Prefer **Annotation polygons (measured)**:
choose the actual sampled region and the boundary from which to scan. For example,
a bed polygon can be the sampled region, with a tumour polygon as the boundary.
**Cell footprint (Classic estimate)** needs no area file, but estimates occupied
area from cell positions; use it only when that approximation is appropriate.
**Area file (CSV)** is an optional alternative for users who already measured
matching zone areas. **Proportion** instead asks what percentage of the cells are
of a particular type and does not need an area.

A p-value tests one comparison. A q-value adjusts across the comparisons selected
in **Correct together**. **q = 1 does not mean the groups are identical**; that comparison
has not passed that correction. Choose the question and
scope before inspecting results; rerunning only the best-looking findings is not
independent validation.

## Several tissue cores in one image

In **Coordinates and units → Separate pieces**, choose the column that identifies
each tissue core or disconnected tissue ROI. IDs may repeat in different images.
Every feature is calculated separately for each image/piece and distance zone;
neighbour searches, dyads, triads and random-label shuffles stay within that piece.

**Advanced: coverage and statistics → Image summary** chooses the median or mean
of those piece values within the independent sample and comparison. Each eligible
piece contributes one value, regardless of its cell count or area. For example,
three core densities of 10, 20 and 90 cells/mm² give a median of 20 or a mean of 40.
This is a summary of core densities, rather than total cells divided by total area.
Pieces without a measurable zone or with too few cells are missing, not zero.
The independent patient count does not increase with the number of cores.

For separate visit results, split comparisons by the visit column. Before/after
change comparisons summarize the cores at each visit first, then subtract the
before value from the after value. A pooled comparison instead summarizes all
eligible pieces across visits together; it does not give each visit equal weight
when their numbers of pieces differ.

The piece ID restricts which cells are analysed together; it does not measure
polygon area or recalculate a saved distance column. Density with **Cell footprint
(Classic estimate)** still uses an estimated area for each piece and zone. Exact
zone areas with separate piece IDs currently require an **Area file (CSV)**;
the direct annotation-polygon area mode does not yet support a piece-ID split.

## Several visits, treatments, or other groupings

One person with before/after tissue has **two observations from one independent
person**. Keep that person's ID the same. A pooled comparison summarizes their
eligible images into one value; it does not treat the two visits as two people.
Use **What to compare → More comparisons** to keep visits separate:

1. Set the main comparison, such as response category A versus B. Leave matching
   off when these categories contain different people.
2. In **Split comparisons by**, select the visit column. Select the treatment
   column too if wanted, then **Combine → Up to two columns together**.
3. Keep **Include pooled comparison** to also report the overall comparison.
   This produces overall, within each visit, within each treatment, and within
   each observed visit × treatment combination. A pooled visit summary is a
   different question from either visit-specific comparison.
4. **Add comparison** lets you test another categorical column. Select a
   particular pair or **All pairs**. For before versus after within the same
   people, tick **Match within samples**. Leave matching off for separate
   treatment groups. Additional comparisons use the same selected splits when
   possible; a column cannot be both fixed to one group and compared in that row.
5. **Also compare changes** adds a direct comparison of change between the main
   groups. Select the repeated-observation column, Before and After. For each
   person, the calculation is **after value − before value**; both observations
   must be measurable. It compares those changes between the main groups.
   It also repeats this comparison within selected treatment strata, but cannot
   split changes by one visit. It does not match individual cells across visits.
6. **Preview comparisons**, then **Screen: Selected**. The preview reports
   independent sample counts before cell/zone coverage checks. Small or
   contradictory groups are skipped with a reason. Each result reports the
   patients actually measured for that feature and zone.

The controls work with your column names; none of these example names is required.
Choose splitting columns for scientific reasons before examining p-values.
The builder supports up to three splitting columns, combinations up to depth
three, five extra comparison columns, and 64 generated comparisons. It does not
silently scan every cell-table column or convert numeric outcomes into arbitrary bins.

For tumor beds, several beds from one person remain repeated observations from
that person. Use a meaningful bed-category column and a matched comparison when
both categories occur in each person. Different individual bed IDs are not
replicated groups; cells within one bed cannot supply independent patients.
The current scanner provides subject summaries, stratified comparisons, paired
tests and two-observation changes, not a general mixed-effects longitudinal model.

Finding a small p-value after treatment and a large p-value before treatment does
not itself show a difference in change. Use the direct change comparison for
that question. It tests an association, not proof of a treatment effect.
[Gelman and Stern explain this distinction](https://doi.org/10.1198/000313006X152649).
Keeping observations from a person together avoids false replication.
[Zimmerman et al. discuss this problem in single-cell studies](https://www.nature.com/articles/s41467-021-21038-1).

**Each feature type · all zones** now includes every generated comparison within
that family. **Whole scan** includes all families, zones and generated comparisons.
**Classic** adjusts within each comparison × family × zone and does not control
searching across comparisons or distances. Untestable declared comparisons remain
in the first two correction scopes. More searching can make q-values larger.

Select a completed **Comparison** in Results to see its plots and sample values.
**All comparisons** and **All FDR findings** download results across the complete
plan. A change plot shows one dot per person; click it to access **Before** and
**After** image buttons. Partial comparisons remain marked Preview with q pending
until the entire plan's correction has been calculated.

Identical cell populations share spatial measurements, including random-label
draws, across comparisons. Group labels do not seed that shared spatial null.
This can change finite-shuffle values relative to an older independently seeded
run; the spatial method is unchanged. Multi-comparison resampling uses one draw
of whole subjects across all questions, retaining their visits and overlapping
subgroup membership. Unlike the single independent-group bootstrap, this draw is
unstratified, so group sizes can vary between repeats. Too-small repeats remain
untestable, with their count reported.

## What the seven features measure

| Feature | Measurement | Important setting |
| --- | --- | --- |
| Density | Target cell count divided by sampled zone area, cells/mm² | Measured polygon areas preferred; Classic footprint estimate is explicitly labelled |
| Proportion | Target count / explicit denominator count × 100 | All zone cells, or named denominator types containing the targets |
| Ratios | Count of A / count of B, optionally log2 | Pseudocount added to both counts; zero leaves undefined ratios missing |
| Diversity | Shannon, Simpson, richness or evenness among selected types | The selected types define this composition; unnamed cells are excluded |
| Neighbour distance | Mean distance from a source cell to its k nearest target cells | Same zone or whole filtered region; directional A→B differs from B→A |
| Pairs | Unordered cell pairs within the chosen distance interval | Count, fraction, Classic random-label Z, or analytical observed/expected; repeated types optional |
| Triplets | Unordered sets of three cells with **all three edges** in range | Its own radius and measurement; repeated types optional |

Choosing target cell types does **not** remove the other cell types from the
proportion denominator or spatial graph. Zero target counts are retained when
the zone has sufficient cells and a valid denominator. Missing zones, absent
neighbour targets and zero denominators stay missing, with reasons in the image
values table. A missing zone is not evidence of zero abundance.

**Growing zones** are nested: 0–50, 0–100, and so on. **Separate rings** are
0–50, >50–100, and so on. Neither identifies an independently validated “best”
distance. Inside uses negative saved distances; outside uses nonnegative ones;
both uses absolute distance. Zero lies on the boundary and is included in the
first zone. With polygons, signed boundary zones are clipped to the selected
sampled region, preserving holes. Buffers approximate circular arcs using 32
segments per quadrant. Polygon coordinates must be image pixels. Saved distance
values may use different distance conventions (e.g. gap-aware distances); polygon
mode deliberately recomputes geometric boundary zones instead of mixing them.

## Density area requirements

The area must describe the **sampled part of the same zone**. Adding cell areas,
using the full slide area for every ring, or estimating tissue from the cells
being counted can bias density. Studio refuses measured-density runs without a denominator. The explicitly selected Classic footprint option estimates area; failed estimates remain missing.

The Classic footprint option uses the same buffered-cell geometry helper as Classic, with a nearest-neighbour percentile of 95 by default, no hole filling, and Classic’s deterministic 10,000-point cap for estimating large areas. Its numerical output agrees with Classic for matching settings. This is not measured tissue area; lost cells, sparse compartments and zone-edge buffers can bias it. The export axis and run notes identify estimated density.

Polygon mode supports one sample/group per image. If a single image contains
multiple samples or you need separately identified pieces, use a measured area
CSV with saved-distance or whole-region mode. The columns are:

```text
sample,group,image,region,zone,area_mm2
sample-01,group-A,image-01,All,0,0.025
sample-01,group-A,image-01,All,1,0.047
```

`zone` starts at zero and follows the entered distances. Use `All` for an unused
group or region. Each key must be unique; every density zone needs an area.
Regions with zero sampled area are unmeasured. Additional cell filters select
the numerator population; ensure the area also reflects any intended restriction
of the sampled tissue. Source files are checked for changes during the run.

### Cell-footprint example: CD4 cells within 0–50 µm

1. Keep cells with saved distances from 0 to 50 µm outside the target, after
   applying the region filters. Treat each image and selected tissue piece separately.
2. Use **all remaining cell types** to estimate area, not only CD4 cells.
   Find the distance from each cell centre to its closest other centre. Select
   the chosen percentile of these distances as one common radius.
3. Draw a circle of that radius around each centre and join the circles. Overlap
   is counted only once. The joined shape's area is the estimated denominator.
   It is **not clipped** to the distance-band boundary or a bed annotation.
4. Divide the number of CD4 cells by that estimated area in mm². For example,
   20 CD4 cells / 0.08 mm² = 250 CD4 cells/mm². Each other selected type uses the
   same area for that piece and zone.
5. Repeat independently for 0–100, 0–150 µm, and so on. These are growing,
   overlapping zones. The footprint and circle radius are re-estimated for each.
   The configured median/mean then summarizes eligible image/piece values within
   each biological sample before comparing groups; this does not pool their areas.

The `99.9` neighbour percentile means a gap near the largest nearest-neighbour
distance, **not 99.9% accuracy** and not 99.9 µm. A larger radius can cover more
empty space and lower the resulting density. The circle construction uses
[Shapely point buffers and unions](https://shapely.readthedocs.io/en/stable/manual.html?highlight=buffer).
For more than 10,000 cells in a piece/zone, Classic uses a fixed random sample of
10,000 positions for this area estimate; the numerator still counts all eligible
CD4 cells. This is another reason the footprint should not be presented as exact
tumor-zone area. Measured polygon intersections provide a geometric area instead.

**No target means no target-relative zone.** When the saved target distances are
missing, that image/piece's zone measurement is missing, not zero. A sample with
no valid zones is omitted from that comparison. If another image/piece from the
same sample has a valid zone, it can still contribute. A valid zone with a positive
area and enough total cells, but **no CD4 cells**, has CD4 density zero. Missing
target distances must remain missing rather than being filled with zero; a numeric
zero distance is interpreted as the target boundary.

## Comparison and interpretation

Images/pieces are summarized with the selected median or mean within each
independent sample. This gives each image/piece equal weight; it is **not** a
pooled-area density. Increasing the number of cells or images does not create more
independent biological replicates.

- **Two independent groups:** two-sided Mann–Whitney with Classic’s SciPy auto
  method by default. An optional permutation rank-sum test handles tied values
  with pooled ranks. Effect = median B − median A. The rank test assesses the
  distributions; it is not exclusively a test of medians. [SciPy Mann–Whitney documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).
- **Matched groups:** paired sign-flip permutation test of mean differences;
  only samples with both groups contribute. The reported effect is the median
  paired B − A difference. Dots/lines show matched observations.
- **Numeric outcome:** permutation Spearman correlation at the sample level.
- **Survival:** continuous-feature Cox model, hazard ratio per SD, with
  proportional-hazards diagnostics. No optimized high/low split. Fits with
  warnings are untested; at least ten samples and five events are required.
- **Describe only:** measurements and figures, without significance claims.

The permutation approach requires exchangeable independent units, or exchangeable
conditions within a matched unit. It does not automatically adjust for treatment,
sampling site, slide, or other confounders. Confounding can explain a difference
even when q is small. Cell types, zones and analysis settings should be chosen
before looking for the most favorable result. [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Choose **Correct together** in Advanced: coverage and statistics:

- **Each feature type · all zones:** Density is corrected separately from
  Proportion, etc.; all selected types and scanned distances within that family
  and every generated comparison are included. This is the starting choice for a new design.
- **Whole scan · all types and zones:** one correction across the entire search.
  Previously saved Studio designs keep this choice until explicitly changed.
- **Each feature type and zone · Classic:** matches Classic's default grouping
  when its pipeline tests each distance separately. It excludes untestable p-values.
  This does **not** control searching across distances, additional comparisons, or combining families.

All three adjusted values are saved as `q_global`, `q_family`, and `q_classic`;
`q` follows the declared choice. Raw p-values do not change with correction scope.
The first two scopes retain untestable planned combinations as p=1 during
correction while displaying their p/q as missing. Choose scope based on the
scientific question before inspecting results, not according to which choice
produces significance. BH is the default; BY is available for arbitrary dependence,
and Bonferroni for family-wise error control. Scanning again
with new choices is another search, not independent validation. When thousands of
tests are planned, the resolution of the chosen permutation count also limits the
smallest detectable adjusted p value. [SciPy multiple-testing documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Stability and uncertainty

Open **What to compare → Stability and uncertainty**. Resampling is optional and
is recorded independently of the spatial random-label shuffles.

- **Sample subsets · Classic stability:** retain a fraction of biological samples
  (default 0.8), without replacement. This reuses Classic's subset sampler.
- **Bootstrap · with replacement:** draw whole biological samples, allowing repeats.
  For a single comparison, independent binary groups keep their group sizes; paired observations stay
  together. Also report a 95% percentile interval for the effect. These are
  pointwise intervals, not simultaneous or selection-adjusted confidence intervals;
  small samples and tied/discrete effects can give unreliable or degenerate intervals.

Both recompute the chosen tests and correction on the resampled sample values,
reusing the spatial measurements. **Recovery** is the fraction of repeats with
q below the cutoff and the original effect direction at that exact distance.
**Nearby recovery** also allows the configured number of adjacent scanned zones.
Untestable repeats count as failures; the testable repeat count is reported.
An originally untestable feature has no recovery score. Bootstrap intervals require
at least 20 and at least 80% testable repeats. They do not replace the original p/q.
The seed makes the resampling reproducible.

Classic calls its without-replacement procedure “bootstrap”; it is a subset
stability check. Studio labels that separately from a bootstrap with replacement.
Unlike Classic's ranked-width/top-feature recovery, Studio checks the declared
feature-by-zone comparisons, requires the same direction, and reports all of them.
Neither procedure is validation on independent patients.
[SciPy describes bootstrap resampling and interval methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

To find **all FDR-significant results**, choose **All feature types**, then
**Findings → FDR significant**. Search and page through the complete results;
the **FDR significant** download contains every qualifying feature and zone.
The publication figure still summarizes a limited number of features for readability.
Clicking another result opens its own sample plot and image examples. Old runs
retain their saved q-values; changing controls does not recalculate them.

Read the three-panel figure as follows:

1. Top: rows are cell types or combinations; columns are scanned zones. Red is
   higher in Group B, blue higher in Group A. The legend identifies the effect
   units. Stars indicate q below the saved cutoff; colour alone is not significance.
2. Bottom left: each dot is one biological sample, with group medians in black.
   The title gives the exact feature, zone, sample counts, p and q. For a motif
   Z-score, positive means more co-occurrence than random labels predict;
   the group p-value asks whether that score differs between groups.
3. Bottom right: the same feature across distance zones. Lines are group medians;
   shading is the middle half of sample values, **not a confidence interval**.
   Growing zones overlap and do not provide independent replications.

## Clickable results and recovery

Click any Results table header to sort all matching findings, then click it again
to reverse the order. Sorting applies before pagination; missing values stay last.
The selected row stays highlighted. Click a row or **Selected feature** to bring
its plot into view in the right Figures panel, reopening that panel if needed.
Background progress does not reopen a panel you have hidden.

**Direction** names the group with the higher estimated value, with a coloured dot
matching that group in the plot. For example, **Higher in PR** or **Higher in CR**.
Treatment comparisons use their own group names. Change comparisons say **Larger
change in …**; survival uses explicitly named event-hazard directions.

**Recovery** is the exact percentage at that distance. The run identifies whether
it used **Sample subsets** or **Bootstrap**. The selected-feature plot also shows
the saved 95% pointwise bootstrap effect interval when available. **Not resampled**
and dashes mean no recovery was saved, rather than zero recovery.

## Ranked stability overview

In **Results**, click **Stability across distances**. It reads the saved tests;
opening this view does not rerun the analysis or change its p/q values. The view
becomes available after the planned comparisons and resampling are complete.

Each row is one feature **and one comparison**; columns are the actual saved
distance zones. Rings show `0–50`, `50–100`, etc.; cumulative zones show `0–50`,
`0–100`, etc. Different outcomes, visits and treatment strata stay separate rows.

- **Blue:** q below the saved cutoff. **Yellow:** raw p < 0.05 only.
  **Pale:** tested but not significant. **Grey:** not tested.
- **Cell text:** the group with the higher estimated value, for example `PR`.
  In a change comparison, it names the group with the larger after-minus-before
  change. For survival, arrows explicitly name higher/lower **event hazard**;
  numeric outcomes use Positive/Negative association. Equal estimated effects
  show `=`; a distribution test can be significant without a median difference.
- **Percentage:** exact recovery at that distance, with the original direction
  and declared q cutoff. `0%` is measured zero recovery; `—` means unavailable.
  Hover or focus a cell for effect, p, q, sample counts, exact/nearby recovery and
  testable repeats. Old runs without resampling display dashes; obtaining recovery
  requires running with resampling enabled.

Rows are ordered by **number of FDR-significant zones in the same direction**,
then **median exact recovery within those zones**, then smallest q and p. Direction
changes are flagged. This is a transparent prioritization rule, not a new test,
measure of biological importance, or independent validation. Several nested
significant zones can reflect the same inner-zone signal. If there are no
FDR-significant zones, the ranking uses the remaining tie-breaks; it does not
promote raw-p findings to FDR discoveries. Choosing a different displayed filter
does not recalculate correction or ranks.

Use the comparison/family filters and search to focus the matrix. Pages contain
10, 20 or 40 rows; **Large view** expands it over the image, and Escape or
**Back to the pane** restores it. Click a tested cell to open its sample plot and
the existing image/Data review controls. PNG, SVG and PDF export the displayed
page; **All rows CSV** exports the complete ranked feature-zone table, irrespective
of the display filters. The existing per-family effect plots remain available.

## Fast pairs and triplets

The new engine builds a radius graph once per image/piece and radius, then counts
all selected types and all zones together. Triplets use sorted adjacency
intersections. Count-only scans do not store triangle instances. Random-label
scans allocate bounded, exact triangle arrays for the shared Classic engine. Counts are exact;
very dense graphs stop with a request to narrow the radius rather than silently
sampling cells. There is no GPU requirement; the random-label option uses CUDA when available and otherwise falls back to CPU. The actual backend, seed and timing are saved per zone.

For a motif of k cells, expected count is the number of geometric motifs multiplied
by its exact probability under uniform random labeling without replacement. The
probability uses the observed counts of each cell type in that zone, including the
multiplicity factor for repeated types. The abundance-adjusted measurement is
`log2((observed + pseudocount) / (expected + pseudocount))`.

That analytical option is **not a permutation Z-score**.

The default **Beyond chance (Classic Z-score)** calls Classic’s
`_permutation_zscore_dyad_triad` directly, keeping positions and type counts fixed
while labels are shuffled within each image/piece/zone. The score is the observed
count minus the shuffled mean, divided by the shuffled standard deviation.
Settings start at 1,000 shuffles and at least 3 observed motifs, matching Classic.
Scores with too few motifs or zero null variance remain missing. Dyads and triads
with matching radii share one null calculation per zone. CPU/GPU streams differ,
so Monte Carlo values need not be bit-identical across devices.

Studio uses the explicitly entered fixed radii and the random-label null. Classic
can also use adaptive radii and a pairwise-preserving triad null; those are different
questions and are not silently enabled here. A complete Classic run is therefore
not automatically identical: selections, radii, null, sample aggregation, zero
handling and correction scope must agree. Studio proportions are displayed as
percentages; Classic stores fractions. Inference about group differences uses
the independent sample measurements, separately from the spatial null. A triplet can be enriched because its pairs are enriched; it is not
proof of an irreducible three-way interaction. Local tissue compartments and
segmentation errors still require review. For context, spatial graph and
random-label enrichment methods are described in [Squidpy](https://doi.org/10.1038/s41592-021-01358-2).

## Results and reproducibility

Results arrive while the job runs. During measurement, previews summarize only
the images finished so far and are labelled **Preview**. Comparisons appear as
feature families finish; adjusted q values remain **Pending** until the complete
scan is available. Reopening the panel reconnects to a running scan. Background
updates do not change the open image or its filters.

The live figure shows one clickable dot per sample/condition. Click a dot, or a
**Higher / Lower** example, to open a representative image, select its context
cells in both the viewer and Data, and zoom to the zone. Polygon-based scans also
show the zone outline. Change the finding dropdown to inspect another feature or
distance. A deliberate selection takes priority over an older response. Unsaved
Data edits must be resolved before selecting a new set of cells.
If the current image belongs to the clicked sample, it stays open. Otherwise the
example closest to that sample's summarized value opens. Matched comparisons show
only samples with measurements in both groups.

Each selected family has PNG, SVG and PDF output: all-zone effect summary,
sample-level dots with group sizes, effect, test name, p and q, and a distance profile
with sample medians and middle-50% bands. Those bands show spread, not confidence
intervals. The displayed feature is selected from the full scan, not an independent
confirmation. The full CSV is authoritative; the
panel lists all tests, with search and pages. Figures show a shortlist, and any
listed finding can open its own live plot. Grey heatmap entries mean unmeasured values.
**Show context cells** explicitly filters the current image to the source cells
and zone; it does not imply every shown cell participates in a motif. Source or
annotation changes invalidate linking. Job progress does not reload the image.

Runs live under the **Output** folder, in `feature_screening/<run>/`
(by default `outputs/spatial_analysis/feature_screening/<run>/`). A project index
keeps older runs discoverable after changing that folder:

- `tables/screening_results_all.csv`: effects, nominal p, adjusted q, sample counts,
  test names and statuses.
- `tables/sample_features.csv`: one value per sample/condition/feature/zone,
  including missing observations.
- `tables/region_features.csv`: per-image/piece values, counts, denominators,
  expected motif counts and exclusion reasons.
- `design.json`: complete settings, calibration, input versions, annotation
  sources, timings and audit counts.
- `figures/`: exported figures; `result.json`: Studio's compact saved display.

For a multi-comparison run, these files live in one `comparison_<id>/` folder per
completed comparison. The parent `design.json` records the full plan and cache
usage; `tables/all_comparisons.csv` and `tables/all_significant_features.csv` contain
the complete jointly corrected search. Skipped questions are retained with missing
p/q and a reason. No source table is modified.

The prior Classic pipeline remains available. Its old saved results are not
reinterpreted as this new engine's measurements.
