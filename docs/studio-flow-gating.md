# Flow gating in Studio

Choose **Phenotyping → Method → Flow gating**. Open its separate window beside the tissue viewer, or choose **Dock below** if popups are blocked. Reload Studio after updating the code.

## Quick workflow

1. Check the images to analyse in Phenotyping and confirm the image and segmentation-label column roles. In an existing Flow workspace, use **Workspace → Sync selected images** to change its scope.
2. Choose the X/Y markers and a plot type. The initial axes use stored values; no transformation is applied automatically.
3. Draw a rectangle, polygon, interval, threshold, or quadrants. Name the population and choose its colour. Open that population to draw a child gate using another marker pair.
4. Click a population name to filter the existing cell layer and Data rows. Its colour is shared with the plots. Overlay checkboxes add comparison curves; they do not change the selected population.
5. Use **Compare** for independently filtered series, or **New plot** for independent axes, filters, and recipes. Right-click a population or plot for actions; the same population actions are available under **More…**. Drag any plot corner to resize it.
6. Enable **Live view**, then choose selected cells, selected regions, the current image, filtered Data, or the viewport. Region/viewport queries use cell centroids in full image pixels; configure the coordinate columns in the Studio launcher when needed. An empty selection stays empty. Pinned reference curves ignore Live view.
7. In **Phenotype labels**, enter `phenotype_flow` or another output column, tick populations, and preview the counts before applying labels. Gate names become text labels. Labels are written to Studio's Parquet journal and appear in Data and the viewer. Source CSV files are unchanged; CSV export remains explicit. Existing labels are preserved by default; overlapping gates are left unassigned unless you explicitly select list-order priority.
8. Right-click a plot and choose **Freeze to layout**. Arrange, resize, duplicate, and label panels; **Style** edits their text and appearance. Export a plot or page as PNG, TIFF, SVG, or PDF. Each export has a JSON manifest. Frozen figures keep their captured values and colours until **Update from current plot** is used.

Use **Clear gate view** in Studio, or **Clear** in the cell-layer Gate row, to restore ordinary colouring. **Plots** restores the workbench after Layout; **Dock below** restores a hidden dock.

The hierarchy indents complete child entries and uses **+ / −** boxes. Axis selectors open all numeric measurements; type to search or use the arrow keys and Enter. **New plot** starts with the active plot's axes, recipe and filters, which can then be changed independently.

Click a population's **colour square** to change it, including **Analysis cells**. Colours save immediately and update live plots and the displayed population in the image; counts and membership stay unchanged. Undo restores the previous colour. Explicit comparison-series colour overrides and frozen publication snapshots retain their own styling.

Click a **2D axis title** to choose its measurement directly. The Y menu also offers **Count**, **Proportion**, **Percent**, **Probability density**, and **Peak normalized**; these switch to a histogram. Choosing a Y measurement switches back to a two-marker plot. Proportion is a fraction per bin, percent is that fraction × 100, and density additionally divides by bin width. Invalid measurements are excluded from the denominator; finite cells outside explicit X limits remain in it. A thin four-sided frame surrounds the plot and exports with it.

## Histogram appearance and panel width

- **Drag a legend entry up or down** to reorder curves. The top entry draws in front; facet and stacked-plot order follows the legend. An insertion marker shows the destination. Alt + arrow keys also reorder a focused entry. Ordinary legend clicks still hide/show individual curves, including independently filtered curves from the same population.
- **Fill** adjusts histogram fill opacity from **0–100%**, leaving solid outlines. Zero gives outlines only; 100% gives opaque fills. It is available above the plot and in Settings. Publication panels have the same control under **Style**.
- **Match Y scales** gives all histogram facets a common range based on the tallest displayed curve, with a small margin above it. It uses the current count/proportion/percentage/density values after optional smoothing. Log Y uses a shared positive range; empty populations remain valid. Turn it off to restore independent scaling. Explicit Y limits apply to all facets.
- Drag the **divider between the population tree and plots** to resize the panels. Double-click resets it. Arrow keys adjust it when focused (Shift makes larger steps). Its width is remembered in this browser; the divider keeps enough space for the plots and adapts to window resizing.

These appearance changes do not rescan cells or change counts, gates, filters or transformations. Plot ordering, fill opacity and matched scales persist across reopening, frozen publication layouts and exports. A frozen panel's styling and legend order can be edited independently of the live plot.

## Split populations by a column

Right-click **Analysis cells** or any child → **Split by…** → choose a column. This is also available under **More…**.

- Text categories and numeric columns with **fewer than 25 distinct values** open a category checklist with exact parent-cell counts. Use search, **All / None**, and checkboxes to choose which values to keep, then **Split**. Only checked values become children; unchecked cells remain in the parent and existing children are preserved. Cancel leaves the hierarchy unchanged. Numeric columns with **25 or more** values open **Threshold / Range** conditions, in stored units. Classification uses all Analysis cells so a continuous marker does not become categorical in a small child.
- Threshold splits use **below / at or above** the cutoff. Range splits use **inside (inclusive) / outside** the interval. Empty or missing values have a separate child. Every child is restricted to its parent; repeat splits reuse existing definitions. Undo reverses a whole split in one step.
- New categorical children use `column_value` names, for example `response_CR` and `response_PR`. This also applies to numeric categories and missing-value groups. Existing shared Studio phenotype colours and the current cluster-review colours are reused based on the original category value. Other labels get a persistent shared palette colour; any swatch remains editable locally. Nested dropdowns show the hierarchy without repeating the column prefix.
- Picking a branch switches the active plot and the image/Data selection. It clears the active plot's previous overlay comparison. Other plots keep their own populations, filters and recipes. Frozen publication panels remain snapshots.

Up to 256 populations are supported in a workspace; a split exceeding this limit is rejected before changing the tree. Filter first or remove unused branches for high-cardinality columns. Low-cardinality filter columns are encoded once and reused for sibling membership checks.

Synthetic split calculation on 22 October 2026: **2.5 million cells, 24 categories** took **1.21 s** to classify/list categories and **0.94 s** to calculate all exact child masks; repeating mask calculation with the encoded column cached took **0.19 s**. These timings exclude reading files, saving memberships, networking and browser rendering.

In **Pan / inspect**, select a gate to expose its resize handles. Rectangle corners, polygon vertices, interval endpoints and threshold/quadrant positions can be dragged. Right-click a gate for **Show cells on image**, **Edit gate**, or **Delete gate**. Delete also works from the keyboard. Every deletion asks for confirmation, including dependent populations when applicable; Undo restores it. Resizing a displayed gate refreshes its image and Data selection; deleting it clears that selection. The existing segmentation layer is reused, and spatial membership is limited to the current image even when local cell labels repeat elsewhere.

## Select, copy, move and delete branches

- **Ctrl-click** (Cmd-click on Mac) population names to select several; **Shift-click** selects a visible row range. Highlighted rows are the action selection. Overlay checkboxes remain independent.
- Drag a selected name onto another population to **copy** the selected branches. **Ctrl-drag copies; Shift-drag moves.** The destination highlights the action. Copy is the default when no modifier is held.
- Right-click the selection for **Duplicate**, **Copy to…**, **Move to…**, or **Delete selected**. The Delete key also works. Deletion asks for confirmation, including children and dependent Boolean populations. One Undo restores the whole operation.
- Copy and move include descendants. Selecting a parent and its child transfers the branch once. Copies receive new IDs; moves retain IDs. Shapes, filters, colours and fitted transformations are preserved, and membership is recalculated within the destination parent. A normalization is not silently refitted.
- Moving into your own descendants or creating a dependency cycle is blocked. When transferring a Boolean gate to another branch, include its referenced populations in the selection. Copied internal references are remapped to the copied populations. Saved dataset labels are unchanged by these tree edits.

Gate corners and vertices have **26-pixel grab targets**. Selected rectangle edges and interval/threshold lines can be dragged along their length; the central handle moves a whole rectangle or polygon. Dragging shows a local outline preview, with one membership update on release. **Escape** cancels the drag.

## Transformation recipes

**Transform** opens a simple named list: **Step 1 → Step 2 → Step 3**. Choose an operation in each step, use **+ Add step**, and expand that step's **Settings** when needed. Steps can be labelled, reordered, disabled, duplicated or removed. **Save transformation** stores the definition without running analysis or changing a plot. There is no reference or target-axis selector in the builder.

**Right-click an X or Y axis title → choose a saved transformation.** The menu lists the complete named library; **Stored values** removes that axis's transformation. X and Y are independent. Editing a saved definition takes effect when it is chosen again; existing plots and gates retain their fitted settings. Histogram X uses the same menu; histogram Y offers frequency and log-scale controls. The saved-transformations selector in the builder reopens an existing definition for editing.

Available operations: fixed/dynamic-cofactor arcsinh, Log10, Log1p, genuine Logicle, Z1, Z2, Normal CDF, NLog, percentile clipping, percentile scaling, Min/Max, robust Z, threshold/ReLU scaling, square root, and linear scale/offset.

- **Z1:** standardize each marker across reference cells, optionally within a chosen metadata column. Population SD (`ddof=0`) is used.
- **Z2:** standardize within each cell after the preceding steps. Its own **Settings → Markers** defaults to the plot's axis markers; **Choose markers** specifies a different set for that step. The axis marker is always included when applying a recipe. Marker choices are resolved and stored with the fitted axis, so browsing other populations does not change them. Multiple Z2 steps may use different marker sets.
- **CDF / NLog:** Studio's Normal CDF and `−ln(max(1−x, epsilon))`, with the same epsilon used by the existing phenotyping implementation.
- **Dynamic arcsinh:** fit a marker-specific cofactor from a chosen percentile of absolute reference values. A zero cofactor is rejected. This is an explicit option, not a biological positivity rule.
- **Normalization scope:** new applications automatically fit on Analysis cells. Fitted parameters are saved with gates. Browsing children or Live regions does not refit them. Existing saved axes/gates retain their original normalization scope.

Clipping and transformations that exclude values report invalid/out-of-range counts. Original measurements are retained. Different recipes can produce different numerical cutoffs; independently normalized groups must not be interpreted as sharing absolute expression units. Normalized gates retain their original reference scope; remove them before changing the workspace's image scope.

## Speed and accuracy

Scatter rendering uses a deterministic display sample (default 20,000 cells; configurable up to 200,000). Gate membership, histograms, density grids, and label assignments use all eligible cells. Styling and layout changes do not rerun numerical analysis.

The backend caches projected columns, fitted recipes, transformed matrices, gate memberships, and histogram results. Editing a gate invalidates only its dependent branches. Repeated Live updates are coalesced; stale responses cannot replace newer plot requests. Large caches are bounded. Processing runs outside the Studio request/event loop.

Synthetic benchmark on this workstation, 22 October 2026: 2.5 million cells, eight markers, 44 images, 136 MB Parquet. These are backend timings, excluding browser rendering and network transfer, not a latency guarantee for every dataset.

| Operation | Observed time |
| --- | ---: |
| Initial projected load and three gates | 2.41 s |
| Cached gate evaluation | 0.001 s |
| Three exact histograms | 0.28 s |
| First eight-marker Arcsinh → Z1 → Z2 → CDF → NLog fit/application | 21.22 s |
| Reuse of fitted recipe and transformed values | 0.005 s |

The first load or a new large recipe can take seconds; repeat interactions reuse that work. Reproduce with `PYTHONPATH=src python scripts/benchmark_flow_gating.py`.

### Saved hierarchies and restart loading

The October 22 transfer update fixes restored plots that still reference deleted populations: the working plot returns to Analysis cells and removes obsolete comparison references. It also avoids pandas copying cached membership arrays into every extracted column, batches the projected Parquet read, preserves Arrow text buffers, and encodes numeric categories before formatting their distinct labels. Plain scatter skips the density-grid calculation; density modes still calculate the full population grid.

Plot-value caches now depend on the plotted populations and their ancestors/Boolean/reference dependencies. Editing an unrelated child does not invalidate an unchanged parent distribution. Current gate outlines and exact saved counts are refreshed separately. Saved counts are accepted only when the dataset revision, image scope and complete gate definitions match. Cache clearing removes those saved count summaries too.

### Cache controls

**Jobs → Analysis cache** reports the current project's generated Flow file count and size. **Clear Flow cache** asks for confirmation and removes only generated plot results and memberships. Saved gates, recipes, Undo history, layout snapshots, labels, source tables and exported figures remain. Active image/Data selections stay usable; analysis files rebuild when needed.

Each workspace automatically retains at most **16 plot results / 64 MiB**, whichever limit is reached first, and **two membership files**. Entries are atomically replaced; older files are removed. Membership history needed by Undo is recomputed from saved gate definitions. Corrupt or unwritable plot caches fall back to calculation. Source changes, selected-image scope, gates, filters, recipes and Live context participate in cache identity.

## Persistence and checks

Workspace definitions, history, recipes, and frozen layouts are stored under `outputs/studio/flow_gating/<table-id>/`. Gate membership is saved as Parquet. Dataset labels use the existing `<table>.labels.parquet` journal. Optimistic revision checks prevent conflicting edits between windows; label writes validate a preview receipt and restore journal backups if a write fails.

Regression coverage includes gate boundaries and missing values, Boolean/nested gates, exact counts, recipe order and reference parameters, repeated segmentation IDs across images, image scope, incremental cache invalidation, label overlaps, text labels in empty numeric columns, rollback, CSV preservation, and export formats. Browser tests cover gate-to-Data/image linking, shared colours, drawn child gates, context menus, plot settings, saved recipes, independent plots, frozen layouts, PDF export, and empty Live selections.

Validation includes the full browser workflow, HTTP startup, marker menus, hierarchy, resize/delete confirmation, image/Data synchronization, new plots, and Jobs cache clearing. Python regressions additionally verify exact membership persistence, restart cache reuse without cell reads, invalidation after label/measurement edits, automatic cache limits, and preservation of gates and research outputs during cleanup.

The split/colour/axis update adds backend coverage for categorical/numeric boundaries, nested membership, missing-value partitions, shared palettes, repeat splits, stale requests, colour-only saves without cell reads, cached colour refresh, and histogram proportion denominators. Its browser check exercises direct axis clicks, histogram switching, the plot frame, nested splits, swatch edits, Undo, independent plots, and image/Data synchronization (`tests/viewer_flow_split_browser.py`).

Transfer/loading regressions cover exact reparented memberships, copied Boolean references, preserved fitted transforms, cycle/limit/stale-request rejection, one-step Undo, zero-copy cache ownership for sliced frames, numeric-category text semantics, saved text overlays in batched reads, dependency-scoped plot caches and restart counts without cell reads. `viewer_flow_transfer_browser.py` exercises multi-selection, modifier drag/drop, bulk deletion, image/Data synchronization, a deleted-population restart, forgiving edge targets, preview/commit and Escape cancellation.

Validation of the transfer/loading update: **71 Python tests** passed across transfer, split, numerical gating, Studio gating, caching, transformations and Data view. **68 browser checks** passed across transfer/selection/resize (22), existing gate controls (21), and the full workflow (25). JavaScript syntax checks also passed.

Validation of the histogram/divider update: **83 browser checks** passed across histogram appearance and divider interactions (37), existing gate controls (21), and the full workflow (25). The new check verifies actual SVG paint order, unchanged exact bins, zero analysis requests for appearance edits, legend hide/show, independent and matched facet ranges, percentages, log axes with empty populations, successive publication-panel edits, SVG/PDF export and reopening. Divider checks cover pointer/keyboard resizing, bounds, reset, hiding and remembered width. JavaScript syntax checks passed. Tests use disposable projects.

Validation of the simplified transformation builder: **45 Python tests** passed across transformations, numerical gating, Studio gating and caches. **83 browser checks** passed across the builder and axis transformation menus (34), direct axis controls/population splits (24), and the full workflow (25). Checks include exact sequential values, independent X/Y application, automatic Analysis-cell fitting, zero analysis requests on recipe save, legacy recipes, versioned edits, individual Z2 marker scopes, changing a Z2 axis marker, histogram frequency controls and reopening. Python checks also verify independent successive Z2 steps and preservation of legacy marker scopes and source measurements.

Validation of the earlier split/colour/axis update: **34 Python tests** in split, numerical gating, Studio gating and cache suites; **70 browser checks** in split/colour/axis interactions (24), existing gate controls (21), and the full workflow (25). Axis tests verify the visible title receives the click before opening its menu. The five split backend tests were repeated after the high-cardinality rejection optimization. Test projects and palettes are disposable.

Validation of the category chooser: **14 Python tests** passed across splits and branch transfers, and **34 browser checks** passed for the split/colour/axis workflow. Checks cover preview without mutation, Cancel, All/None, search, exact parent counts, selected-only children, missing versus literal “Missing”, low-cardinality numeric choices, invalid/empty selections, stale previews, preserved existing children and colours, Undo, and image/Data synchronization. JavaScript syntax checks also passed. All datasets and palettes used in these checks are disposable.

The same **14 Python tests and 34 browser checks** passed after adding column prefixes to new categorical gate names. They verify prefixed nested and missing-value labels, category-based colour reuse, copied branches, and hierarchy dropdowns without duplicate column prefixes.

This change passed **52 focused Python tests** across `test_studio_flow_cache`, `test_studio_flow_gating`, `test_studio_startup_summaries`, `test_studio_phenotyping_inputs`, `test_phenotyping_images`, and `test_studio_single_image_phenotyping`. A final targeted run repeated the 23 cache/gating/summary tests after the naming-cache refinement. **70 browser checks** passed across Flow workflow (25), gate controls/synchronization (21), startup (7), cache management (7), deferred Phenotyping summaries (3), and Phenotyping column roles (7). Browser checks use isolated synthetic projects; the full-table timing runs use disposable caches and unchanged research measurements.

HTTP startup regression (22 October 2026): the original localhost browser tests missed the absence of `crypto.randomUUID` on ordinary network HTTP addresses. That caused startup to fail before drawing the controls. IDs now use `crypto.getRandomValues`, which is available on those origins. The detached page has an initial loading message, recoverable errors, and versioned script loading. Startup no longer duplicates the table request in the parent window. The full browser workflow now runs on the non-secure `http://flow-gating.test` origin, where `randomUUID` is verified absent; 25 workflow checks passed. A separate startup test reproduces the old failure and checks recovery states and initial Live synchronization.

Optional dependencies are in the `flow` extra (`pip install '.[flow]'`). Logicle requires FlowUtils; PDF and page rasterization require CairoSVG. Both were installed and tested in this workstation's `spacemap` environment.

## Current limits

- Long first-time calculations show an Updating state; they do not yet provide cancellable Jobs-panel entries.
- Spatial region membership uses centroids, not polygon–cell overlap area. A plot's individual scatter point does not yet select a single cell in the tissue viewer; saved population selection and viewer-to-plot Live selection are supported.
- Arbitrary grouped/ordered recipes do not always have a single reversible boundary. Such gates are shown only in matching transformation coordinates; membership always uses their saved coordinates.
- Publication pages are editable snapshots. Shared-legend automation, linked axis groups, marker-by-group layout templates, and batch export of every page are not yet included. ScatterGL and 3D traces can remain rasterized inside SVG/PDF.
- Gate assignment uses saved populations over the checked-image scope. Live view is an inspection filter and does not silently change the label-writing scope.
