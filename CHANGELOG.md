# Changelog

Notable changes to SPACEMAP, newest first. This file is written by hand and
describes what each release means for users.

For the complete record of every commit — who made it and why — see
[docs/LEDGER.md](docs/LEDGER.md), which is generated from the Git history.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Studio release (0.4.0)

- Make Studio the primary documented workflow, with a persistent image viewer,
  coordinated analysis panels, marker tiles, editable data tables and figures.
- Add configurable CPU/GPU execution where supported, background job timers,
  informative progress, process termination and protection against concurrent
  writes to the same project across machines.
- Extend manual-reference and consensus phenotyping, model reuse, neighbourhood
  review, consistent colours and exploratory TLS candidate detection/editing.
- Preserve the published academic research license, Python 3.10 compatibility,
  native-library installation fixes and optional AI dependencies.
- Include a workflow illustration, Studio guide, release validation notes and
  broader checks for machine-specific paths and missing packaged viewer assets.

- Add a resizable **Marker tiles** drawer with selected/all marker views, native
  zoom presets, cursor tracking, pinned cell inspection, and a floating/docked
  layout. Tiles follow the image colours and ranges, including Cell Top markers.
  Save panel preferences with the image view; load only visible cards.
- Add **Ask AI** to the cell suggestion card: send measured features, model and
  dataset phenotype context, and labelled marker crops for advisory review.
  Text-only models receive measurements with a clear image-support notice.
- Recover MR-ML controls when the image session finishes loading after the panel;
  retain entered settings during layer refreshes and explain checked-only training.

- Allow MR-ML to train from a single reference column, retain existing reference
  labels and predict remaining cells using markers plus optional morphology.
  Add named prediction, numeric score and low-confidence filtered columns in
  Studio; save results in Parquet and write/export them to CSV from Jobs or
  Classic. Existing ensemble MR-ML remains available.
- Add **Top markers / Number** to Cells and maps, using the selected cell-marker
  transformations. Ctrl-click selects multiple cells for averaged marker display,
  combined prediction scores and label counts, with explicit group assignment.

- Exclude annotation history directories and cached snapshots from viewer layer
  discovery and the project GeoJSON picker. Old directory entries now return a
  clear error instead of crashing tile requests; saved History stays intact.

- Add a native **100%** viewer zoom button and percentage presets; remove the
  approximate magnification section. Improve image detail with
  lossless native-resolution tiles, finer pyramid selection and crisp enlarged
  pixels. Optional smoothing is saved per image; overview navigation keeps
  compressed tiles without colour subsampling. Full-resolution figure exports
  also request lossless image tiles.
- Fix thin lines between opaque PNG image tiles at fractional zoom by explicitly
  declaring their opacity; preserve transparency in registration checkerboards
  and annotation overlays.

- Fix bulk polygon labels disappearing while the list refreshes. **Result label**
  names the next marker detection; **Apply label** labels checked existing polygons,
  with automatic saving and Undo.
- Add **Subtract annotations**: create an editable layer such as
  `Stroma = Tissue − Tumor − Epithelia` from multiple saved layers or labels.
  Preserve holes, source polygons, History and the equation's JSON provenance;
  reuse the result in Classification and Export.

- Align Marker regions with the shared Work within/Scale and polygon review layout;
  remove its stray null placeholder and preserve separate named result layers.
- Reopen generated regions and classifications as editable layers. Reuse their named
  classes directly; save subclassifications into child layers without overwriting parents.
- Add checked-layer GeoJSON export with names, colours, source revisions and History.
  Record derived-layer relationships in project JSON for later model development.

- Studio Tissue detection now saves directly, and reviewed artifact subtraction
  updates the selected working annotation with Undo and History. Both Artifact
  tabs retain Work within, shared with Classification.
- Classification adds explicit resolution and channel/feature selection, compact
  naming controls, reversible cluster merges, linked heatmap ordering/colours,
  and reviewable AI suggestions using the parent annotation and full profiles.
  Review settings and provenance are saved per image in project JSON files.
- The launcher automatically selects another port when one is occupied. A running
  Studio serves a frontend matching its backend, preventing mixed-version controls.

Interface review for publication: every setting saved with the project,
one layout for all Phenotyping methods, one terminal format, faster reads in
Tabs 7 and 9, and a Z1 option for Resolve. A built-in image viewer. Grouped
by tab.

### SPACEMAP Studio

- Add a persistent **Save polygons** button at the bottom of Tissue, with a
  save confirmation. Show an amber T for unsaved previews, waiting edits or
  failed saves; restore green after saving. Keep background saving in place.
- Preview necks from the current Tissue layer, including unsaved edits, with
  checks for changed images or saved polygons. Do not restore discarded pieces
  from an old review when rerunning Pieces; use History to restore them explicitly.
- Click the mouse wheel over the image to fit it to the viewer.
- Extend Shift/Alt brushing to the drawing tools and temporary tissue previews.
  Add adjustable Split width and Merge gap, keeping exact saved outlines and Undo.
- Move Pieces into Tissue with an optional neck-review panel, a width slider,
  Classic's two-side size limits, and per-neck approve/reject. Link preview
  visibility to Tissue and keep explanations in help tips. Align piece names
  beside their checkboxes in both polygon lists.

- Allow multi-delete of Tissue preview polygons from either polygon list or
  with Delete; keep Undo and add **Save polygons** to save the reviewed result
  without running detection again. Preserve full outline detail when saving.
- Save Annotation settings automatically and finish saving before switching
  stages or steps. Keep newer edits made while an earlier save is in progress.
- Show Tissue **Smallest** in µm², like **Fill holes**, and enforce it on final
  polygons at every scale. Convert old pixel thresholds to their physical area.
- Prevent delayed Segmentation and Annotation replies from restoring controls
  or previews over a different step or image.
- Preserve detailed tissue outlines in previews and editable layers; a drawing
  point limit no longer turns complex boundaries into coarse triangles.
- Add **Fill holes** in Tissue settings: fill enclosed holes smaller than a
  chosen area in µm². Zero keeps all holes. Preview and saved detection agree.
- Keep image controls first in Layers, fold other layers into groups, and add
  All/Shown list filters. Group saved Outputs separately. Fix marker alignment
  when Segmentation's N/M buttons, Dye and Exposure are shown together.
- Make Annotation tissue and artifacts editable with a brush, polygon
  checkboxes, select-all, Delete selected, Undo and saved History. Detect
  artifacts first, review them, then subtract them from tissue. Rerunning
  tissue detection asks before replacing the current layer.
- Fix tissue previews reporting zero pieces when the detected outline contains
  a collection of polygons. Keep holes and count connected pieces correctly.
- Explain interrupted polygon saves and failed loads, retain waiting edits,
  and add Reload saved and Retry save controls in both polygon panels.
- Stop duplicate Tissue, Artifacts and Marker regions layers accumulating when
  images reopen. List temporary Tissue previews in the same layer, marked as
  unsaved. Show each polygon file's location and allow clearing saved polygons
  with Undo and History available.
- Add a viewer-centred workspace with all ten analysis steps, disk-based work
  lists and links to classic SPACEMAP. Images load first; hidden annotations
  load only when requested.
- Add checked viewer/assistant actions, name matching, Undo and command logs.
  Common requests work without a model; local AI supports other wording and
  labelled image descriptions.
- Place Live RESOLVE beside the image, share classic options, edit phenotype
  panels and run all ROIs in a background job with progress, Stop and logs.
- Add exact marker-rule selections, click/drawing refinement, sampled marker
  comparisons, reference assignment with guarded Undo and a project rule library.
- Add saved reference-model training/comparison and classic MR-ML background
  jobs. New prediction columns can be displayed without reopening the image.
- Restore blue marker bars with ordered transformation choices, editable
  phenotype menus, background label saving, review flags and fast saved-model
  suggestions in the clicked-cell summary.

### Image viewer (new)

See [docs/viewer.md](docs/viewer.md).

- **Open image viewer** on every page opens raw QPTIFF scans, registered
  images and ROIs at full resolution, zooming through the pyramid levels
  stored in the files. Markers show with their real names; each has its own
  colour and display range.
- Tab results are drawn on top, each with its own opacity:
  - cell outlines or filled cells from Tab 5, coloured by any column
    (phenotypes, neighbourhoods, marker intensities, distances), in the Tab 9
    phenotype colours, with a legend that hides a class when you click it;
  - tissue annotations in their class colours;
  - the registered H&E in true colour.
- Click a cell to see its labels and highest markers. **Save image** saves
  the screen as a PNG with a scale bar. **Save full resolution** saves the
  same area at full image resolution for figures (up to 8,192 px on the long
  side; larger areas are saved at the largest resolution that fits).
- **Label cells (Manual Reference):** click cells or drag a box to name them
  into `phenotype_ref1` (or any column), with undo. Labels are grouped and saved in the background,
  as Tab 7's Manual Reference saves them. Tabs 7, 8 and 9 list them under
  "Saved but not yet written into the CSV". A new label column appears in
  every tab's column lists straight away, before it is written into the CSV.
- Each tab opens the viewer on its own results: Tabs 7 and 9 colour cells by
  the newest phenotype column, and Tab 8 by the neighbourhood column.
- **Drawing:** rectangle, polygon and freehand tools draw shapes on the
  image. Shapes can be named, coloured, imported from the project's GeoJSON
  files or from a file (QuPath exports), downloaded and undone. They are
  saved as GeoJSON as you draw.
- **Working beside SPACEMAP:** **Separate window** opens the viewer in a
  window of its own that follows the images opened from the tabs and
  refreshes when a tab writes new columns. **Full screen**, **Minimize** and
  **Restore image viewer** are added, and each image's view is remembered.
- **Faster and sturdier:** an image opens in under a second the first time
  (it was 8-9 s on PDAC and INNATE ROIs) and at once afterwards. A new ROI's
  cell outlines are prepared in the background while the image already
  shows. Viewer links, and an open separate window, keep working after
  SPACEMAP restarts.
- **Marker sets:** save ticked markers with their colours, ranges and names
  under a name ("T cells", "B cells") and switch between them with one click,
  on any image of the project.
- The separate window can open any of the project's images itself; changing a
  marker or its range no longer zooms out or flashes the image (the new
  version is loaded underneath and swapped in when ready).
- **Density maps** of chosen phenotypes (or of all cells, or the local
  average of a column), made as in Tab 9, in one colour or a colour map.
  **One map per group** overlays several groups to show proximity and
  overlap. Tab 8 triad instances can be mapped the same way.
- **Layer order** (arrows on every layer) and **other GeoJSON files** (from
  your computer or anywhere in the project) as extra layers.
- **Cell card:** top markers ranked by Z1 (compared with the ROI's cells),
  Z2 (within the cell), Arcsinh or Raw, as RESOLVE scores them.
- **Fixed:** the cells of some ROIs did not show. A Tab 3 ROI-boundary file
  (`data/annotations/new/<sample>.geojson`) had been taken for the cell
  segmentation. Only Tab 5's folders are searched now, and a file must hold
  numbered cells.
- **Marker histograms:** click a marker for its histogram (whole image or
  inside a drawn shape) and drag its display range, as in QuPath's Brightness
  & Contrast. Marker colours, ranges and new names are saved with the
  project (`outputs/viewer/display/`), where other steps can read them.
- `run_interface.sh` and the `spacemap` command now start
  `src/spacemap_gui/server.py`. It serves the interface and the viewer's image
  data on one port, so nothing extra has to be forwarded on a cluster.
  Streamlit 1.58 or newer is required (the lock file already pins 1.58.0).
- `import spacemap` loads its modules only when first used. It no longer
  loads PyTorch, Cellpose and SCIMAP at start-up, so SPACEMAP starts faster:
  the import took 58 s on a cold node and now takes 1-5 s.

### Installation and release packaging

- Standardized the supported runtime on Python 3.10 across package metadata,
  Conda, Docker and CI; added the native OpenSlide library to setup instructions.
- Made AI installation optional in the default Conda environment and added a
  first-project walkthrough, CPU setup and remote-server instructions.
- Included shared marker defaults and the compiled heatmap selector in release
  packages. The selector now loads its plotting library locally and ships
  third-party license notices.
- Fixed the command version to use installed package metadata, added an
  OpenSlide native-library check, and kept the project registry and active
  session under the configured application state root.
- Added package-content checks and corrected the change-ledger commit procedure.
- Disabled the unused Xvfb pytest plugin installed by SCIMAP, so headless tests
  do not try to start a virtual display.

### Licensing

- Replaced the MIT template in the private repository with the UT Southwestern
  academic-research-only license. Commercial use and redistribution, including
  any use or redistribution by a for-profit entity, are prohibited.
- Updated the README, package metadata, citation metadata and release guidance
  to describe the same terms. Third-party dependencies and models keep their
  own licenses.

### All tabs

- **Fixed: the cell visualizer showed image channels in the wrong colours.** It
  read OME colours byte-shifted (a blue DAPI came out green); it now reads them as
  OME defines them (red, green, blue, opacity).
- **Fixed: switching to a moved project froze SPACEMAP.** If a project's
  folder had been moved, choosing it in the sidebar loaded nothing and the
  page restarted itself endlessly (full CPU, Ctrl+C slow to respond). Now the
  sidebar looks for the project next to your other projects and remembers its
  new place. If it is not found, it shows an error and stays on the current
  project. Projects whose folder is missing are marked "(folder not found)".
- **Fixed: the project file could grow to tens of GB.** Tab 8's
  neighbourhood result (a per-cell table held inside a session value) was
  saved into the project file, which reached 44 GB and made SPACEMAP take
  minutes and all the node's memory to start. The size check measured a
  table by its short printed summary, not by the one-row-per-cell JSON
  actually written. Now:
  - every setting is measured as it will be written, tables nested in lists
    and dicts included, and anything data-sized is left out;
  - Tab 8's result holders are never saved;
  - a project file is never written above 32 MB;
  - a project file that has already grown is cleaned when it is opened:
    it is streamed, not loaded, every setting is kept, and the original is
    kept beside it as `<project>.json.oversized-<date>` for you to delete.
- **Settings are saved with the project.** A setting is saved only when its
  control has a key that belongs to a tab. The controls below had none, so
  their values were lost on reload; they now save and restore. A new test
  (`tests/test_widget_keys_persist.py`) scans every tab and fails if a setting
  control could not be saved.
- **One run log for every tab.** Each tab has a single box, **Record run
  log**, in the same place under the page guidance, on by default and switched
  back on whenever a project is loaded. It replaces the separate "Create
  Processing Log (CSV)", "Create Log", "Create reproducibility processing log"
  (one per Tab 7 method) and "Record process events" boxes. While it is on,
  every button pressed on the tab -- and any page update that takes longer than
  2 s -- adds a row to `<project>/logs/run_log.csv` with the start and end
  time, duration, CPU time and cores used, peak RAM, peak GPU memory, the tab's
  settings and which of them changed since the previous run, and the machine,
  Python, SPACEMAP version and code revision (`spacemap_gui/run_log.py`). The
  Leiden background worker writes its own row for the whole run. The detail
  logs a tab already kept (per-sample registration logs, per-method phenotyping
  logs) follow the same box. A "Run log" panel at the foot of each page shows
  the tab's recent runs.
- **Saving phenotypes no longer rewrites the CSV.** "Save All Assignments to
  Disk" in Leiden, SOM, Vortex, Resolve and SCIMAP stores only the changed
  columns beside the unified table (`<table>.csv.pending/`,
  `spacemap/pending_columns.py`), in seconds instead of minutes. Tabs 7, 8 and 9
  read them at once. The file panel in those tabs shows what is waiting and a
  **Write pending changes into the CSV** button that writes columns and Manual
  Reference labels in one pass, in its own process, so it finishes even if the
  interface is closed or restarted; the old CSV stays until the new one is
  complete, and nothing saved is lost if the interface crashes. If the CSV is
  rewritten by something else first, the saved columns are not applied by row
  position: the panel offers **Re-apply by cell ID** or **Discard**. Saving and
  then writing produces the same CSV as the old full rewrite (tested).
- **One terminal format.** Every tab now prints
  `2026-09-22 14:03:11 | Tab 7 Phenotyping | Resolve | message`, with
  `Started:` / `Finished: ... (3.0 s)` lines around timed steps and `ERROR:` /
  `WARNING:` prefixes (`spacemap_gui/terminal.py`). About 200 `print` calls and
  twenty separate log helpers in ten formats now go through it. The existing
  switches still apply: `SPACEMAP_CONSOLE_LOGS`, `SPACEMAP_TAB_LOGS`,
  `SPACEMAP_PERF_LOGS`, `SPACEMAP_APP_LOGS`, `SPACEMAP_TRACE`.
- **No emoji.** About 190 emoji removed from labels, buttons, messages and
  expander titles. Icons, where they help, use Material icons
  (`:material/warning:`).
- **Dead code removed.** 154 unused imports and 14 repeated local imports.
  `spacemap/segmentation_cellpose.py` no longer imports `palom`, so Cellpose
  segmentation no longer requires it.

### Tab 1 — Registration

- **Any CODEX image can be paired with any H&E image by hand.** Manual Pair
  Overrides listed only images whose names followed the naming rule, so an H&E
  named e.g. `S7 HE scan.tif` could not be chosen. Now every image in the two
  folders is listed; the rule still makes the automatic pairs. Two CODEX images
  that would give the same sample name (and so the same merged image) are flagged
  and left unpaired until one is skipped or the rule changed. Studio's
  Registration step uses the same pairing and the same saved choices.
- **The registered image keeps all of the CODEX metadata.** Before, only each
  channel's name, colour and fluorophore reached the merged image. Now every field
  the CODEX file records is kept, per channel (marker, exposure, filters and their
  bands) and for the slide (slide ID, study, objective, camera, scan profile, ...),
  as OME key/value annotations that QuPath, OMERO and Bio-Formats show. Exposure
  times (exact, from the source) and excitation/emission wavelengths also go in
  OME's own fields, and the original CODEX and H&E file names are recorded. Checked
  with Bio-Formats: valid OME-XML, one pyramidal image.
- **Fixed: channel colours were saved as fully transparent.** OME colours carry an
  opacity byte; SPACEMAP wrote 0 (transparent), so a viewer that honours it could
  hide the channel. Colours are now opaque; red, green and blue are unchanged.
- **The coarse alignment's matched points are saved as a figure**
  (`<sample>_he_registered_feature_matches.png`, beside the registered H&E), as
  the Registration notebook showed them. Figures are now closed after each pair,
  so a long batch no longer accumulates them in memory.

- CODEX/H&E filename prefix and suffix used for pairing are now saved.
- Each pair has **View CODEX**, **View H&E** and **View registered result**
  buttons. They open the full-resolution scans in the image viewer, next to
  the thumbnails. The registered result is shown with the H&E over the CODEX
  so the alignment can be checked at any zoom.

### Tab 2 — Background Subtraction

- **A background-removed image keeps all of its parent image's metadata**
  (exposure, wavelengths, acquisition notes), and notes which levels or blank
  channels were subtracted.

- Blank-subtraction input source and selected image are now saved.

### Tab 3 — ROI Extraction

- **Draw ROIs in the image viewer** draws, names and edits ROIs on the
  full-resolution image. They are saved straight into the ROI file that
  **Extract ROIs** reads, and each ROI's name becomes its output name.
- **Use names and colours from the image viewer** copies markers renamed or
  recoloured in the viewer into the channel table, and so into the extracted
  images.
- **Each ROI crop keeps all of its parent image's metadata**: exposure times,
  excitation/emission wavelengths and every acquisition note (per channel and for
  the slide), plus which image and box it was cut from. A marker renamed or
  recoloured for the crop keeps the new name and colour and notes its name in the
  parent. Checked with Bio-Formats: valid OME-XML, one pyramidal image.
- **Fixed: ROI crops were read as one image per plane by tifffile-based tools**
  (napari, Python scripts): a second, tifffile-added description made every page
  its own series. QuPath (Bio-Formats) was not affected. The same fix applies to
  background-removed images and CODEX-to-CODEX registration.
- Exposure is written in OME's own field (`Plane/@ExposureTime`) instead of a
  non-standard `Channel/@Exposure`; SPACEMAP reads either.

- **Fixed:** marker names showed as empty boxes and colours as black when
  blanks had been saved in the project (seen for INNATE's P11). Black was also
  written into the extracted ROI files. A blank name, or black where the image
  gives a colour, is now filled from the image again.
- **Fixed:** TMA core-detection settings (core size, sensitivity, row gap, row
  label format, name prefix) were never saved: their keys belonged to no tab.
- **Fixed:** numbered TMA rows produced duplicate core names — row 1 column 12
  and row 11 column 2 were both `R112`. Numbered rows are now `R1_C12` and
  `R11_C2`. Lettered rows (`A1`, `B12`) are unchanged up to 26 rows.
- Plain-language help for every TMA setting.

### Tab 5 — Segmentation

- **An annotation file can be chosen by hand for an image or cell table** whose
  name does not match its annotation (`tab4_annot_geojson_matches` for the
  segmentation mask, `seg_tag_geojson_matches` for cell tagging). The matching
  tables show the chosen file; Studio sets these choices.
- The segmentation region (whole image, annotation classes or drawn mask) and
  the spatial columns are set up by one function, and cell tagging runs from
  saved settings, so Studio runs them exactly as this tab does.

- Segmentation QC plot settings (colours, bins, scale, plot type, variables)
  are now saved.

### Tab 6 — Batch Correction

- **New default: cyCombine's own method, exact.** `spacemap/cycombine.py`
  ports cyCombine's R code (its table workflow, as in the released 0.2
  versions): each marker z-scored within each batch only to group cells; a
  kohonen SOM (8 x 8, 10 passes, seed 473) with R's own random numbers, so
  every cell lands in the same group as in R; sva's ComBat within each group;
  each marker kept within the group's measured range; the covariate used per
  group only when cyCombine would use it. Checked against R (cyCombine 0.3.0,
  kohonen 3.0.13, sva 3.48.0): the same group for every cell and values equal
  to 15 decimal places. Where R gives missing values (a group of two cells
  from two batches), the measured values are kept. Several covariates chosen
  together act as one (cyCombine takes one).
- **Fixed: Monotonic correction widened each batch.** It scaled each batch
  to the spread of all cells in an SOM group, which also holds the batch
  differences being removed. It now scales to the group's spread within
  batches, so a cell type's spread stays as measured.
- **Renamed:** Per-cluster ComBat is now "Earlier cyCombine approximation"
  (a different SOM and ComBat library); keep it only to reproduce an earlier
  run. Projects that saved a method keep it until you choose another.
- **Removed dead code:** three QC functions in
  `spacemap/batch_correction_utils.py` were defined twice, identically; the
  unused first copies are gone.

### Tab 7 — Phenotyping

- **One layout for every method.** Leiden, SOM, Vortex, Resolve, SCIMAP, MR ML
  and Consensus ML now open the same way: title, one-line summary, a collapsed
  "How this method works" guide that names the matching notebook, then
  numbered sections — 1. Input File, 2. Marker Selection, 3. Configuration,
  the method's own steps, Run — followed by "Results & Phenotype Assignment"
  and "Visual Validation of Representative Cells". Sub-Phenotyping and Tools
  use the same opening.
- **New: Resolve "Z1 score (no threshold files)".** For projects without
  threshold tables. Each raw marker is ArcSinh-transformed and turned into a
  Z1 score within each batch; a marker is positive when its Z1 score is above
  the cutoff (default 1.5, adjustable). Z2, CDF and NLog are switched off in
  this mode because they would change what the cutoff means. The same rule is
  available to notebooks as `spacemap.ph_thres_utils.apply_z1_positivity`.
- Resolve's Threshold Method help no longer describes five methods that were
  removed (Li, Yen, Mean, IsoData, Minimum).
- **Settings now saved:** the "Import Columns from External CSV" tool (anchors,
  shift tolerance, columns to transfer, renames).
- **Faster:** batch-ID lists, phenotype-name scans and column-import anchors
  read single columns through the Parquet fast copy instead of parsing the
  whole CSV (minutes to about a second on a cohort-sized table).
- **Removed dead code:** two SCIMAP helper functions in
  `spacemap/ph_scimap_utils.py` were defined twice with different signatures;
  only the second copy was ever used, and the first is gone.
- Plain-language help for Leiden's GPU option and the UMAP subsample size.
- Leiden, SOM and Vortex column pickers, and Tab 0's project fields, no
  longer log Streamlit's "value set twice" warning on every render.
- **Leiden while a run is in progress:** the page says when the run started and
  that the Run button returns when it finishes, and offers **Stop this run**.
  The previous run's terminal output is hidden during a new run and labelled
  as the previous run otherwise. Leiden run times were checked against earlier
  runs and are unchanged (about 22 min for 8.6 M cells: 2.7 min reading the
  CSV, 11.5 min clustering the 941 k-cell sample).
- Removed leftover "Professional ... synchronized with notebook" captions.

### Tab 8 — Spatial Analysis

- **Faster: cell-cell interactions.** The permutation test counts neighbouring
  pairs with one count per phenotype pair instead of one update per pair; the
  z-scores and p-values are unchanged (tested against the earlier counting).
- **Removed a control that did nothing:** Density Panel 2's "Zone area buffer
  percentile" slider was never read — Panel 2 uses Panel 1's tissue area.

### Tab 9 — Visualization

- **Fixed:** Paper Figures cached the unified table by path only, so an edited
  or re-exported table kept showing the old rows until restart. The cache now
  follows the file. Column lookups use the Parquet fast copy; the whole-table
  read that feeds model training keeps the pandas reader.

### Notebooks

- **Fixed `Step5_batch_correction.ipynb`:** cell 6 contained a block spliced
  from the loading loop, so the cell could not run, and the advertised
  `'column'` batch-ID strategy always failed. The block is back in the loading
  loop and the summary prints the batch column.
- **Fixed `Step6C_phenotype_threshold.ipynb`:** 59 source lines had lost their
  line breaks, so `PHENOTYPE_EXCLUSIONS` and the exclusion step were silently
  merged into comments and never ran. Exclusion markers apply again, matching
  the Resolve tab's "Must not have" column. Added `USE_Z1_POSITIVITY` /
  `Z1_CUTOFF`, matching the new Resolve option.

## [0.3.0] — 2026-09-19

First public release.

### Added

- `spacemap` command-line entry point that launches the interface and applies
  the numeric thread caps the pipeline needs.
- A user-writable state root (`spacemap.userpaths`): settings, caches, logs and
  models live under `~/.spacemap` (or `$SPACEMAP_HOME`) instead of inside the
  installed package. Existing in-package state is migrated on first run.
- A model catalog (`spacemap.model_catalog`) that makes model choice data
  rather than code, with a built-in list and a user-editable override.
- In-GUI model installation: browse a catalog, download from Hugging Face with
  a progress bar, import weights from a folder on an offline machine, add a
  model the catalog does not know about, and remove models to reclaim space.
- Hardware detection (`spacemap.hardware`) that reports what the machine can
  run and chooses a load precision.
- Optional dependency extras: `ai`, `gguf`, `deepcell`, `notebook`, `all`.
- GGUF backend, so a quantized model can be served through llama.cpp on a
  machine with no GPU.
- A change ledger (`docs/LEDGER.md`) generated from the Git history, a
  `commit-msg` hook requiring every commit to record why it was made, and a CI
  check that the ledger is current.
- CI covering lint, tests on Python 3.10–3.12, a no-extras install that proves
  the pipeline works without the AI packages, a guard against absolute paths
  from a contributor's machine, and a check that notebook outputs are stripped.
- Contributor documentation, issue and pull-request templates, MIT license and
  citation metadata.

### Changed

- **The AI assistant is optional and off by default.** Its page is not
  registered and its modules are not imported until it is switched on in
  Tab 0. `SPACEMAP_ASSISTANT=off` disables it for an entire installation.
- The local model server accepts any model directory instead of one fixed
  model, on a configurable port, and handles both multimodal and text-only
  models.
- `deepcell` moved out of the base dependencies into an extra, so a normal
  install no longer pulls TensorFlow.
- The Slurm integration is site-neutral: partitions are ranked by what Slurm
  reports, with optional site preferences, and the feature hides itself when
  Slurm is not installed.
- Notebooks build every path from a `DATA_DIR` configuration cell, and their
  saved outputs are stripped.
- The test suite is self-contained: it creates its own temporary state root and
  fixtures, and needs no research data, no models and no GPU.

### Fixed

- **The local model server no longer refuses to start without a GPU.** It
  loads on CPU in float32 instead. Quantized precisions still require CUDA and
  now say so plainly rather than failing.
- The model server ships inside the package, so it is present in a `pip`
  install and not only in a git checkout.
- The assistant's path guard no longer assumes a fixed directory depth, so it
  is correct for an installed package as well as an editable checkout.

### Removed

- Institution-specific paths, cluster names, account identifiers and internal
  planning documents.
- Committed interface state containing one developer's absolute paths and
  project names.

[Unreleased]: https://github.com/AguileraLab/SPACEMAP/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AguileraLab/SPACEMAP/releases/tag/v0.3.0
