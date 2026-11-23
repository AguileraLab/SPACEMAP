# SPACEMAP Studio: how every step is designed

Current standing requirements: [AGENTS.md](../AGENTS.md) and the
[Studio skill](../skills/spacemap-studio/SKILL.md). These include the user's
latest feedback on concise panels, shared Cell data, stable viewer choices,
startup performance, and publication figures.

For future changes, use: “Follow `AGENTS.md`, `skills/spacemap-studio/SKILL.md`
and `docs/studio-design.md`.” The first file routes project work to the skill;
the skill keeps standing interaction requirements; this document records the
panel design and workflow decisions.

This is the design standard for Studio steps. Registration and Background already follow it.
When a step is built or redesigned, follow this file first. Only add what that step truly needs,
and list it under "What each step adds" at the end.

## 1. Principles

- **Simple buttons.** One row per action at the bottom of the step: a short label ("Detect", "Export",
  "Extract"), then **This image** and **Selected**. No extra modes, pop-ups or duplicated controls.
- **The viewer does the looking.** Clicking a name opens the image in the viewer. Items that belong
  with it (its annotation, its H&E) appear in Layers. Use the viewer's own tools (Auto, Layers,
  drawing, colours) instead of copying them into the step.
- **Work first, files later.** Anything the step computes (levels, shapes, cores, previews) is kept in
  the project and shown on the image. Files are written only when the user presses the step's
  write button (Export, Extract, Register…). Exception: editable annotation layers (Tissue,
  artifacts, regions, classes) save as you work, because every change can be undone (Ctrl+Z)
  and every saved version stays in History.
- **Batch everywhere.** Every action works on the open image and on the ticked images.
- **Nothing is lost.** Keep originals or retained backups when the user explicitly chooses to
  update an existing output. New images keep all of their parent's metadata (pyramid,
  tiles, channel names and colours, dye, exposure, pixel size).
- **Always say what is happening.** Loading shows "Loading…" at the top, and jobs show "N jobs
  running". After every action a note near the buttons says what happened in plain words. Long
  work runs as a job, and the Jobs tab opens.
- **Same files as classic.** Reuse the classic tab's functions, file formats and saved keys, so
  Studio and classic stay interchangeable.
- **A tidy Layers list.** Keep images and markers first; fold other layers into
  Cells and maps and Annotations. Folding only tidies controls. A checkbox shows
  or hides the result on the image. All/Shown filters the list. Outputs lists
  saved files separately, in folded groups. Remember the user's open groups.

## 2. Page layout (top to bottom)

### A. Folders

- One text box per folder, holding the full path. The folder can be inside or outside the project.
- Typing a path and pressing Enter saves it, refreshes the list, and shows a note.
- Label each box with a single word (Images, Annotations, Thresholds, Output…). Its "?" tip says
  what goes there and what file names are written.
- When a step takes several input folders of one kind, add a **(+)** button next to the first box.
  The columns then read "CODEX (1)", "CODEX (2)"; with only one folder, keep the plain name.

### B. The list

- A row of count chips above the table, for example "12 images · 3 with blanks · 5 removed".
- The table columns, in order:
  1. a tick box, headed by a **select-all box**. Ticking the header box ticks every row; clearing
     it clears every row; it shows half-ticked when only some rows are ticked. Never add a
     "Pick all" button.
  2. the name, as a link that opens the image;
  3. the step's matches, as dropdowns (annotation file, class, H&E…). A dropdown's first entry
     is "By name: <match>" or "Auto: <match>"; the user can always pick a file by hand.
  4. the state, as chips (for example "Detected", "CSV", "Image", "Registered", "Not run");
  5. step-specific buttons (for example Matches), when needed.
- Marks after the name: ⚑ for a note (blank cycles, a missing match) and ✎ when the settings were
  edited. Each mark has a tooltip explaining it.
- Clicking a row opens or selects it; the row is highlighted. Controls inside the row do not open
  it.
- Below the table, one line each for leftovers, such as files that match nothing.
- **The list follows running jobs.** Update the working row's blue "Working…" chip in place. Progress must not
  rediscover inputs, erase cell counts or rebuild directory controls, image rows, or marker previews. Keep focus,
  selections and scroll position. Refresh saved result badges when a relevant job finishes, even if another job still runs.
  Spatial uses `followJobs(current, [job kinds], refreshResults, {progress:updateWorking})`; older callers still use a
  throttled reload callback. Progress messages identify the image (`"<image>: …"` or `"<image> · region <id>: …"`).

### C. The open image's settings

- The step's method row (method, "in": whole image or annotation class) sits once, above the
  markers, never repeated below.
- Then the markers table (section 3).
- Then any step-specific lists (the ROIs of the image, for example).
- Changes save to the project as the user edits, with a short note ("Markers saved for X.").

### D. The bottom bar

- One row per action: `Label: [options] This image | Selected`, with a "?" tip in plain words.
- The options are tick boxes for what to write. Background's Export, for example, has
  "Thresholds (CSV)" and "Image (OME-TIFF)".
- For "all", the user ticks the header box, then presses Selected; there is no separate All
  button. Registration still has "Register all"; leave it as it is.
- A status line under the rows.

## 3. The markers table (the same in every step)

Each marker is one row, with columns in this order:

| Column | What it is |
|---|---|
| Keep | A tick box, with a select-all box on top. Blank cycles are unticked by default. |
| Step-specific left columns | For example, the Anchor ◉ in Registration. |
| Colour | A colour swatch. It starts with the file's own colour; a changed colour is saved as the user's choice. |
| Name | An editable box that holds the name itself (no separate rename box). Typing over it renames the marker; emptying it, or typing the file's name back, undoes the rename. The tooltip shows the name in the file. |
| Dye | From the file (OME Fluor or QPTIFF), or "-". |
| Exposure | Exposure time in ms, from the file, or "-". |
| Step-specific right columns | For example Blank, the Level and Auto in Background. |

- The same final name twice among kept markers gives a warning line: "Same name twice: … Untick one
  or rename it."
- Choices are saved per image, keyed by the image's file name.

## 4. Detect, check, then write

This is the pattern for anything computed.

1. **Detect** (This image | Selected):
   - For the open image, use the viewer's own tool, so the result shows at once.
   - For the other ticked images, run a job.
   - The method and "in" come from the step's method row. The class comes from each image's row.
   - Results are kept in the project only. For example, levels are the image's display settings in
     `outputs/viewer/display/<image>.json`.
2. **Check** on the image in the viewer, and adjust by hand if needed.
3. **Write or Export** (This image | Selected), with tick boxes for the kinds of output.
   - The open image uses its values as shown; the others use the values kept in the project.
   - Ask before replacing earlier files.

## 5. Words and help

- Use plain, lay words: no emojis, no internal terms.
- Button labels are short verbs.
- Every section and action has a "?" tip. It says what happens, what is written, where, and what
  stays unchanged.
- Error messages say what to do next ("Tick the images first.", "Open an image first.").

## 6. How it is built (for whoever writes the code)

- **Server module:** `src/spacemap_gui/studio/<step>.py` holds the following. Heavy work
  reuses the `spacemap` functions.
  - the saved keys (reuse classic keys);
  - `FOLDER_KEYS`;
  - `all_folders`;
  - the list plan (`images_plan` / `plan`);
  - the per-image settings (`image_markers` / `pair_markers`);
  - `save_step`;
  - the job functions.
- **Commands** go in `schemas.COMMANDS` and `commands.py`. **Jobs** go in `jobs.KINDS` and
  `tasks.<kind>`. **Routes** go in `viewer/api.py`, as `GET` for the plan and `POST` for changes.
- **The browser never sends file paths:** only names. The server resolves each name inside the
  step's folders.
- **Page code:** `TOOLS.<tab>` in `viewer/frontend/studio.js`. Use the shared helpers:
  - layout: `section`, `field`, `folderBox`, `fill`;
  - controls: `choice`, `pickAll` (the select-all box), `chip`, `help`;
  - the bottom bar: `current.runBar`;
  - server calls: `runCommand` and the tracked `get` / `post`, which drive the busy indicator.
  - Viewer panels live in `viewer/frontend/viewer/*.js`. They are shared with the step through
    hooks passed to `startSpacemapStudio`.
- **Code changes need a restart.** Running Studio processes serve a snapshot of their frontend matching their loaded backend. Restart or launch a new instance to load code changes; live source edits must not expose unfinished UI routes. The launcher chooses another local port when the requested/default port is busy and leaves existing processes running.
- **Tests:** add a unit test (`tests/test_studio_<step>.py`) and a browser check
  (`tests/viewer_studio_<step>_browser.py`). Also update `viewer_studio_browser.py` when the step
  list changes.

## 7. Checklist for a new or redesigned step

- [ ] Folders on top, each a full-path box; Enter saves, refreshes the list and shows a note.
- [ ] List with a select-all header box, name links, match dropdowns, state chips, and ⚑ / ✎ marks.
- [ ] One method row; markers table as in section 3.
- [ ] Detect, check, then write; the results are kept in the project until written.
- [ ] Bottom rows `Label: [options] This image | Selected`, each with a "?" tip and a status line.
- [ ] Outputs keep the parent's metadata; ask before replacing; originals untouched.
- [ ] Busy indicator, notes and jobs; plain words.
- [ ] The list follows running jobs (`followJobs`, `workingChip`); job messages start with `"<row name>: "`.
- [ ] Classic keys and file formats reused.
- [ ] Unit test and browser check pass.

## 8. What each step adds

| Step | Unique items |
|---|---|
| Project | No folders, list or bottom rows of its own. Top to bottom: **Project** (name, folder, last saved, notes), **Projects** (the list classic SPACEMAP keeps, latest saved first; click to switch; open a folder; New project, folded), **Samples** (each sample's state in every step; click a dot or step number to go there), **Folders** and **Model library** (both folded). Settings that belong to a step stay in that step; restore points, the audit log and caches stay in classic Tab 0. Opening a project in Studio switches Studio at once; classic SPACEMAP opens the same project the next time its page runs, saving its open project first. |
| Registration | CODEX and H&E pairs by naming rule or by hand; extra CODEX folders (+); pixel size (µm/px) per pair; Anchor marker; Compare (overlay, rotate, scale, move, Quick align); Matches figure; right-click zoom menu (percentages, 100% = one image pixel per screen pixel). |
| Background | Annotation and Class per image; blank cycles paired by dye (⚑); level per marker, shown as its minimum; Before / After; Export as CSV and/or OME-TIFF. |
| Regions of interest | ROI file per image ("By name" or a GeoJSON picked in the row); the open image's ROIs drawn and named on the image; TMA core settings and Detect cores (the cores become the ROIs, to check before extracting); Extract with "Whole box" (default: outside the shape set to 0); crops keep all of the parent's metadata. |
| Annotation | Stage tabs (Tissue with necks and pieces, Artifacts, Classification, Marker regions, Zones, Export); editable polygon layers that save as you work, with Ctrl+Z and History; brush (Shift adds, Alt erases, wheel sizes); Work within (follows the image into later stages); artifacts trained from red/green examples, reviewed, then subtracted from tissue; Classification heatmap with merge, names and Ask AI, whose saved polygons also give Zones its domains; Export with Merge same names (on by default). Artifacts: `Find: [Replace saved] This image \| Selected` with the chosen model (each ticked image's polygons saved as found; saved ones kept unless Replace saved), and Subtract from tissue `Selected`; **Train on selected** beside Train model makes one model from each ticked image's saved (hand-fixed) polygons and green examples, on top of the chosen model's other images (images never reviewed are left out). Classification: **Classes from** Each image its own (default) or a shared class model; `Shared classes: Selected` fits one set of classes on a fair sample of each ticked image (20,000 pixels each, each image scaled within its own Work within), chooses it and classifies every ticked image; names, colours, merges and exclusions belong to the model (name once); Scale each image or each slide (the slide is the name before _roi: one scaling per marker for all its images; marker intensity only); per-image checks (pixels fitting no class, classes whose profile differs); `Save polygons: This image \| Selected`. Subtract annotations has `Selected`: the same names on each ticked image (a minus name it lacks takes nothing away; no base name: skipped). The stage letters in the list are buttons: a letter opens its image at that stage and shows the stage's polygons (A: every artifact kind). A stage left while loading cannot change the bottom bar of the stage shown. Details in section 10; the work log is [annotation-log-2026-09-29](codex/annotation-log-2026-09-29.md). |
| Segmentation | Folders: Images, Output (cell tables), **Outlines folder** (folded; geojson/ in Output unless changed), **Models** (any folder of Cellpose models; nothing is copied). **One list**: each image has **one** Area, Whole image or one of its annotations, grouped like the Annotation lists (this image's names, then each imported file's); a name present in several files requires one explicit source file; the Area header sets all ticked images (or all). Green "N cells" when segmented. Markers table with Nucleus and Border. **Active model** (Cellpose, Trained in this project, Models folder). Settings save as they change; GPU batch defaults to Auto. Training: tiles folded and scrolling, Tile outlines / Saved cells; cell controls in Cells and maps. Bottom: `Segment: [Again] [Then tag cells] This image \| Selected \| Selected polygons`; Then tag cells runs the Cell tagging step on the new tables in the same job (classic Tab 5's inline tagging is not used). After a job the open image reopens whenever its outlines changed. |
| Cell tagging | **Folders and tables**: Tables as **One per image** (CSV or Parquet per image; Folder box, Follow Segmentation on by default) or **One merged table** (Folder box, the table, the column naming each cell's image; rows are its images); Annotations (the Annotation step's exports, and an Annotations folder box for files matched by name); Output box (tagged tables keep their format). Folders are typed as full paths, inside the project or not. Out-of-date tables are flagged. Tables list (tick, image name opens it with its cells, annotation per row, green Tagged). Each table's annotations default to the Annotation step's export of its image (final names), else a file matched by name or picked. **Tissue pieces**: the parent annotation, with Inside / Piece / Size columns, units µm or pixels, edge cells. **Annotations**: per annotation, Inside / Piece / Size / Distance, each a tick plus its column name (renamed in place). Distances stay inside the cell's tissue piece (straight line, on by default); each annotation's distance is measured on its own, so only the piece's edge blocks the line. **Beds**: two or more annotations; each cell goes to the nearest, within its piece; one bed column with the winner's name, plus <name>_bed / <name>_bed_id. **Preview**: in memory, optionally cells in view only; colour by any new column (distances and sizes as a colour scale). `Tag: [N columns] This image \| Selected`. Classic's engine; Studio renames and converts afterwards (`tag_columns.py`). |
| Batch correction | **Folders and tables** first: full-path Images / Cell tables / Output. Images define the list; match CSV/Parquet tables by ROI identity (tagging suffixes and cache twins never create ROIs). Missing/ambiguous matches stay visible and disabled. Select-all, ROI image links, a correction-unit column (Slide ID when correcting between slides), parent slides from the ROI step, green saved-result checks (independent of table adoption), stale/unverified states. **Samples** follows the list, before **Settings**. Related controls share responsive rows. Settings autosave; normalization/grid/passes/seed/cofactor are in Advanced. Cell filters and post-correction QC are editable and share classic settings; invalid percentile pairs cannot save or run. Output name and optional **Save CSV too** sit with the output folder. Both formats use a unique run name; Jobs retains later CSV export. Changed-input notices describe the old result and explicitly allow a new run. `Correct: Selected` requires two technical units. Samples supports clinical CSV/TSV imports and manual columns. Imports choose both the source key and the dataset key: ROI, Slide ID, cell-table filename, Samples column, or cell-table column. Dataset joins apply per cell without multiplying rows; metadata follows cells into new unified outputs. Legacy tables without input provenance get an informational note, not a timestamp-based claim that their cells are stale. Samples, Combined table and Distribution checks start folded; diagnostics load only on opening. Pooled EMD/relative MAD are labelled diagnostics, not scientific pass/fail scores. Each result keeps settings, input stamps and metadata in its JSON manifest. See [scientific review](batch-correction-review-2026-10-09.md). |
| Spatial analysis | **Images**: the combined table's images (tick with a select-all box, name opens the image, cells, chip "Neighbourhoods" when a neighbourhood run covers it, else "Not run"). Methods as tabs: **Neighbourhoods** (settings save as they change; groups found, named and shown; Where in each image: stacked bars linked to the image), **Cell–cell interactions** (heatmap beside the image), **Density** (cells per mm² per image with classic's virtual-tissue outline, drawn on the image when its row is clicked; a group's density map on the open image; each group by distance bands from an annotation, a group or band filtered on the image), **Triads** (classic's test as a job; each image's triangles drawn on it with the triad density map; classic's saved triads below), **Feature screening** (Cells and regions → Zones → Features → Compare → Results; seven independently selectable families, explicit units and density areas, generic sample/outcome columns, sample-level comparisons and correction across the full feature-by-zone search; exact graph counts, per-family figures and CSV/PNG/SVG/PDF exports; explicit context-cell review and saved runs; see `studio-feature-screening-guide.md`). **Gland types** shares its engine and saved settings with Classic: generic compartment/bed mappings, size-aware composition fitting, interior/exterior profiles in the right figure dock, exact gland/bed selection in the viewer and Data, and independent-sample comparisons across visits/strata. Results use linked columns without rewriting source measurements; see `studio-gland-types-guide.md`. Bottom rows `Fit:`, `Find:`, `Test:`, `Density:`, `By distance:`, `Find triads:`, `Screen:` with This image \| Selected; all images: tick the header box. A run on some images computes on their cells only; result columns still cover every row. |
| Visualization | Tabs: **Figure** (images list with ticks; title, legend, scale bar, size; `Save figure: This image \| Selected`, each ticked image opened in turn), **Heatmap** (mean markers per group; a row shows its cells), **Proportions** (stacked bars per image; a segment opens that image showing that group), **UMAP** (a fair sample per image, made as a job and kept; colour by any group column; a dragged box filters those cells on the open image, a click opens the cell), **Method agreement** (the F1 paper figures' measures: accuracy, precision/recall/F1 per phenotype, the confusion table; a reference phenotype shows its cells coloured by the method). |

Segmentation has one **Active model** selector at the top of Settings, used by
**Test on this view** and every Segment action. Testing sits in the same section.
Training shows its **Training model** as a read-only value: it predicts correction
tiles and supplies the next training round's weights. Saved versions live in a
folded **Model history**, with validation scores and **Use for segmentation**.
That action immediately updates Active model and the history's Active indicator;
it does not run segmentation or replace existing cell outlines. New training
versions appear when the job completes without silently activating themselves.
Model and parameter saves are serialized; failed activation reports an error and
must never claim success. Epochs and training provenance stay in folded settings
or help. Refresh model metadata without rescanning images or loading weights.

Training cell outlines belong to **Cells and maps**, beside the original or
imported segmentation. Use one **Training cells** panel with **Current tile**
(editable predictions) and **Saved cells** (kept training and validation cells),
independent visibility, shared Appearance, and **Tile outlines** nearby. These
cells do not become tissue annotations or acquire annotation export controls.
Keep the polygon correction tools and **Keep tile** workflow intact. Left-side
visibility controls and Layers must stay synchronized. Saved cells use a compact
overlay, cached until the image or saved tiles change; late results cannot undo
Hide, Discard, or image navigation. Shared **Cell data** applies to measurement
layers, not temporary tile geometry. Put optional top-marker, marker-set and
scale controls in folded **Cell inspection** below the linked cell layers.

Segmentation's **Nucleus** and **Border** columns use checkboxes. Multiple nuclear
markers are combined by the brightest value at each pixel; border markers are
combined separately. At least one nuclear marker stays selected. Without border
markers, the nuclear signal supplies both model inputs. Settings and Layers'
N/M buttons stay synchronized. Preview, whole-image runs, masked runs and training
tiles share the same input construction; saved marker measurements stay separate.
Explain the combination in the Markers help tooltip, not a paragraph in the form.
Tissue/ROI areas restrict the predicted masks, not the input pixels before
normalization. Zeroing the background first changes percentile scaling and can
erase the contrast of bright-background channels, including H&E. Retain the
original crop for inference, then clip output cells to the selected area and its
excluded holes before measuring or exporting them.
For segmentation, a repeated annotation name across different GeoJSON files
must resolve to one explicit file. Group the existing Area dropdown by file
when the name is ambiguous; retain the selected file through navigation. Shared
Local/Imported names remain usable when they identify a unique source. Do not
infer the analysis source from visibility or combine hidden/older imports.
Job logs print the absolute source GeoJSON path, annotation name, and generated
mask path. Ambiguous legacy selections fail before previous results are moved.
Keep each job's area snapshot in its job folder so later edits or runs cannot
replace the boundary it used. Missing, invalid, empty or out-of-image areas
stop the run rather than enabling whole-image segmentation.

Proposed H&E extension (not yet connected to segmentation): derive **H&E nuclei**
from the original HE_Red/HE_Green/HE_Blue planes using hematoxylin stain separation.
Raw RGB maxima are not a nuclear signal. Use same-section images with local
nuclear alignment checked; exclude missing/padded pixels, keep input scaling
independent of viewer brightness, and put DAPI and hematoxylin on comparable
scales before fusion. Compare DAPI, H&E and combined inputs on reviewed crops
before adopting a model. Reuse the Nucleus list and help tooltip.

The future combined Clean Tissue workflow and retained training provenance are
tracked in [the Clean Tissue roadmap](codex/clean-tissue-roadmap.md).

## 9. Viewer: image detail and zoom

- **Copy preview** sits beside Save image and in the image right-click menu.
  It copies the same PNG view as Save image, including visible overlays and the
  scale bar, without toolbar buttons or other interface panels. HTTP or denied
  clipboard access opens that image for the browser's native **Copy Image**;
  show success only after a confirmed clipboard write.
- Keep a visible **100%** control: one source pixel per logical viewer pixel,
  independent of µm calibration. Show the current percentage in its tooltip
  and the right-click menu. Use only percentage zoom presets; the user requested
  removal of the approximate magnification section and × presets.
- Select image pyramid levels using zoom and display density. Use lossless
  native-resolution tiles and lossless tiles for full-resolution figure export.
  Keep compressed, cached overview tiles for fast navigation, without JPEG colour
  subsampling. Preserve display RGB values at native resolution; never apply
  artificial sharpening or modify source/analysis pixels to claim better detail.
- Default to crisp native/enlarged pixels and filtered reductions. Offer
  **Smooth enlarged pixels** in the right-click menu and persist it per image
  with the other project JSON display settings. Keep all existing drawing tools.
- Verify a known pixel pattern on the browser canvas at 100%, including a
  high-density display, in addition to checking tile dimensions or source levels.
  Also check actual canvas pixels across tile junctions at fractional zoom and
  pan positions, with opacity and registration checkerboards. Declare image
  tiles opaque explicitly; PNG encoding alone does not imply transparency.

### Clicks, selection and right-click

- A plain click replaces the open cell or region at once, before anything is loaded, so Enter and other shortcuts
  never act on what was shown before. Ctrl-click adds to a selection.
- When polygons overlap (marker regions over the tissue, say), a click or a selection outline goes to the layer the
  current stage works on, then to the smallest polygon.
- With the Move tool, right-click opens the menu (name, zoom, delete). It deletes at once only while a drawing tool
  is on. Artifact polygons have their own small menu (Delete, Mark as).
- A step's own layers (its tissue, artifacts, regions, ROIs) show when its image opens; Cells and other extras stay
  off until turned on. The layer a stage draws into is shown.

### Cells and data in Layers (7 October 2026)

- **Cells and maps** lists one block per cell layer (Studio's segmentation, an imported one, Voronoi cells or circles
  from a table; a tick hides it, its title is the folder it comes from), each with one form:
  **Data**: only tables holding these very cells (the same cell numbers at the same places as the segmentation's own
  table, checked on a sample; else the same count). A table made from another segmentation is not offered, and a
  "hidden" chip says which and why. **Show**: outlines, filled, or both; outline colour; opacity. **Colour by**:
  the data's columns as Groups (text: a colour each, click to hide) or Values (numbers: colour map and range).
  **Filter**: keep ticked groups or a range of values; several add up; cells filtered out are not drawn at all;
  "N of M cells shown". Then the density maps as rows and Import cells / Cell data.
- An image always shows the **Segmentation step's latest outlines**: the viewer reads the folders as saved now (not
  as they were when the project opened). Their **own tables come first** in Colour by ("Tagged cells (Cell
  tagging)" when newer, then "This segmentation's cells"); a combined table made before the cells is listed as
  "made before these cells: may not match", since its cell numbers belong to the earlier segmentation.
- **Cell data** (CSV, TSV, Parquet; dropped, typed or from Cells and maps > Cell data…): a dialog guesses the x/y
  columns and their units (µm is divided by the pixel size), the column naming each row's image and this image's
  value, and the cell number column. Show as: this image's cells (linked by cell number, or by the cell each position
  falls in, when the numbers come from another run), circles (size in µm) or Voronoi cells (reach in µm, for tables
  without a segmentation); Within an annotation (grouped list) keeps only its rows and cuts the shapes at its edge.
  It becomes a cell layer (all Cells controls: outlines or filled, Colour by any column, ranges, groups); its row says
  how many rows it shows, exports the circles or Voronoi cells as GeoJSON, and redraws with a new size, reach or
  Within. Kept per image in outputs/studio/cell_data/<image-key>/ (the rows used, the shapes, an index); the table
  itself is never changed. Linked tables join the image's own cells' table list as "Data: <file>".
- **Data tab** (the panel below the image: Marker tiles | Data): a cell layer's table (linked to its cells) or a
  table opened here (button, drop → "Open in the data viewer", path). Rows of this image (a merged table's image
  column and value), pages of 200, every column (scroll sideways), sorting by a header click, and a filter button
  per column: groups get tick boxes with their counts (a search box, All, None), numbers a histogram to drag a range
  across plus From / To; counts are over the rows the other filters keep, and each change applies at once. A column
  picker hides columns. Linked both ways: a row outlines its cell (and pans to it); a cell clicked on the image selects
  its row (Follow the image); "Selected cells" keeps the rows of the cells picked on the image; with "Image follows
  filters" (on) the cell layer draws only the rows' cells. Double-click edits a value; Ctrl+C copies the selected rows; Ctrl+V pastes down
  a column; edits stay with the table until Save (a copy, or into the file with the original kept in previous/;
  a cell layer's table is saved as a copy in outputs/studio/data_edits). Export: these rows, shown or all columns, as
  CSV. The table is never read whole: its Parquet form is read by column and by row group (the image column first,
  then only the columns a filter, the sort or the page needs). A large CSV opens once its Parquet twin is made (in
  the background, the first time); a 2.8 million row, 138 column table opens in about 5 s and pages in under 0.3 s.
- **Drops are sorted by what they are**: a cell table opens the Cell data dialog; a GeoJSON of numbered polygons
  (cell outlines) is offered as cells first (or as annotations); any other GeoJSON becomes annotations; an image
  opens or lies over the open one.
- **Bring files in**: drop a GeoJSON or an image on the image, or type a file or folder path in Layers (a remote
  node too). A GeoJSON becomes an editable annotation layer (imported copy). An image (QPTIFF, OME-TIFF, TIFF, SVS;
  PNG/JPG/BMP become a TIFF in outputs/studio/dropped) opens, or, at the open image's size, lies over it as its own
  layer (tick, opacity, markers, ×).


### Annotations in Layers: by source, then by class (23 October 2026)

The user chose flat names. Layers > Annotations has one **Working** block for this
image's drawn/generated annotations, one **Copy** block per editable imported file,
and one **Linked** block per read-only GeoJSON. Each source has a single header:
disclosure, visibility checkbox, name, source kind, count where available, help,
and removal. Full paths and provenance belong in help. Identically named imports
remain separate, with stable file identifiers when needed; never sum their counts
into one apparent file or hide a distinct linked file because its name matches.
Do not delete or merge duplicate files automatically. A linked feature may contain
multiple polygons, so its feature count need not match an editable copy's polygon
count. Visibility does not select the segmentation Area.

Linked class names and counts load independently of image and cell tiles; the
legend endpoint reads only the registered GeoJSON. Requests share one result,
time out after 60 seconds, and show an error with Retry on failure. Transparent
or removed layers, and density layers without a source, do not hold subsequent
overlays in the opening queue. These rules also apply to the classic viewer.

**Top markers**, its count and help share the Annotations header and appear only
while expanded. Their interaction must not fold the group. Tight source margins
and class-row padding preserve fonts and controls; colour fills the checkbox
height. Long class lists have search and visibility for matching classes, keeping
the visibility of nonmatching classes. Within a source, use **one row per class**:

- A row: show/hide (hides only those polygons), a colour (saved in the file as `colour`, used on the image and in
  the export), the polygon count, an **Export** tick, and ⧉ Duplicate. Provenance
  is in help rather than an extra text row.
- Annotation names are indented beneath the file; polygon rows are indented again.
  **Split** (or expansion) exposes individual polygons and their tools; **Group**
  folds them back into fast class display. Both buttons share the annotation header.
  Group only within the same source, annotation name and colour, maintaining source
  drawing order. This batches display paths, not saved geometries: boundaries,
  holes, IDs and labels remain intact and editable. Same-class overlaps use one
  shared fill rather than accumulating opacity. Training-cell and artifact review
  overlays retain their independent rendering and controls.
- **Use as Tissue…** sits beside the expanded annotation title and disappears when
  folded. Sort, label and minimum-size controls share a line in wide docks. Polygon
  rows have minimal padding; status, Reload saved, Saved file, Download and History
  share a wrapping footer. Fonts and ordinary control sizes remain unchanged.
- Reuse geometry paths, bounds, areas and centres while their geometry object is
  unchanged. Retain SVG nodes across selection updates, and limit drawing to the
  viewport plus a small margin. Offscreen polygons stay in the source for selection,
  measurements and full-resolution export. Editing, reload and undo replace geometry
  and invalidate its cached calculations. Image exports honour the same grouping,
  class visibility and holes; GeoJSON downloads preserve the original polygons.
- Artifacts (Blood clots, Image quality) split into Artifact and Good tissue (training), each with its own Export
  tick; training is unticked by default.
- **Rename**: select polygons on the image (click, Ctrl-click, Select outline), right-click, choose a name or New
  name: they move to that row. **Duplicate** copies a row's polygons as a new layer under a new name.
- Read-only files (zones, classic outputs, GeoJSON added from disk) are their own blocks, listed once each.
  Their classes, opacity and file actions sit directly under that one header.
- **Delete** (×): a name row deletes its polygons (Ctrl+Z and History bring them back); a copied, imported or
  generated layer holding only that name leaves the project, its file moved to `outputs/studio/deleted/<image-key>/`.
  Tissue and artifact layers keep their (empty) place. An imported file's block has its own × for the whole file.
  A read-only file's block × takes it off the image's list (`outputs/studio/annotation_hidden/<image-key>.json`);
  the file stays on disk.
- The naming rule is one, in two places kept in step: `annotationName` (viewer/regions.js) and
  `annotation_names.name_of`: a hand-given name wins; tissue pieces belong to the tissue; artifact polygons to
  their artifact; otherwise the class. Keys are `<source>::<name>::<role>`.
  These shared export keys are separate from each source block's file identity.
- **Export** (Annotation step) lists the same names with the same ticks (one saved choice,
  `outputs/studio/annotation_export/<image-key>.json`) and writes one GeoJSON: each polygon named and coloured as
  its row, same names merged.
- **Names for every image**: a name given by hand (Label selected polygons, right click, a rename or recolour) is kept
  for the project with its colour (`outputs/studio/annotation_names.json`, newest first). Every image offers these
  names, with colour swatches, in the right-click menu and the label box. A name's colour on the image wins; else the
  project's; else one from the palette.
- **Export** has three parts, top to bottom:
  1. **Imported files**: Import GeoJSON files (or drop several on the image, or **From a folder**: every GeoJSON in it,
     as a job `import_annotations`; a file already imported for its image, by name, is skipped; an earlier export can be
     imported on purpose).
     **Manage imported files** lists them across all images (filter, tick, Remove ticked: Studio's copies go to
     outputs/studio/deleted; a project file found by name is only taken off the image). Segmentation shows the same
     box under its folders. Each goes to the image whose name it
     contains, read as words and numbers (P11-ROI1 old.geojson to P11_roi_1, never to P11_roi_10), else the one
     clearly closest name; a file matching none waits in the list until its image is chosen. Studio keeps its own
     copy; the files are never changed. The images list shows an **I** letter before T: green when the image has an
     imported file; a click opens it on Export with the imported layers shown.
  2. **What to export**: one set of rules for every image, keyed `<origin>:<name>` (origin *local*: made in Studio or
     a file in the project's outputs; *imported*: a GeoJSON brought in or found outside the outputs). Scan images
     lists the annotations of the ticked images (all when none is ticked; the open image before a scan). **Keep
     inside** (one rule): every exported polygon is cut to it, except pieces renamed by hand in the Studio tissue
     layer. **Drop pieces smaller than** (µm², 0 off): pieces of Keep inside left after Subtract below it go, with
     every polygon in them, as Remove small pieces does in Tissue. Per row: **Export**, **Subtract** (cut out of all the others; a subtracted annotation that is also
     exported stays whole), an export name and colour (the file only). Layers' Export tick is the same rule. Training
     examples have no rule. Nothing is exported until ticked.
  3. **File names**: <prefix><image name><suffix>.geojson in the folder (the image's own name when both are empty).
  The bottom bar exports This image (also downloaded) or Ticked images (a job): each image applies the rules to the
  annotations it has; an image without any ticked annotation or without the Keep inside one is listed as not
  exported. Settings: `annot_studio_export` (prefix, suffix, folder, merge, inside, rules).
- **Delete results…** (beside the stage tabs): the shown stage's saved files for the open image or the ticked ones are
  set aside in `outputs/studio/deleted/<image-key>/`; the letter turns grey. Classification also drops its saved run
  (settings kept); Export also its named file.

### Working inside annotations (every step)

Image-layer controls use five bordered sections: **Image display**, **Marker
sets**, **Auto levels**, **Markers**, and **Histogram**. Keep percentage values
beside opacity sliders and align the marker list's visibility, colour, name,
segmentation roles, and Low/High columns. Search and select-all operate on the
matching markers without clearing other choices. Advanced display settings start
closed; explanatory text belongs in `?` help.

The pixel histogram starts on **Whole image**, independently of Auto levels.
An explicit histogram region is retained for that marker during inspection.
Show raw range, median, pixel count and resolution as compact values. Changing
the histogram area or axis does not change the display levels or measurements;
dragging its handles explicitly changes display levels. Changing only the Y axis
reuses the measured histogram. Load annotation classes on demand.

Keep these sections close together: remove empty padding before shrinking text
or controls. Marker rows use compact vertical spacing, with colour fills as tall
as their checkboxes. Preserve input, button and histogram sizes.

Wherever a step or the viewer asks where to work ("in", "Measure in", "Segment"):

- **Whole image** first.
- Then grouped as Layers: the image's name, then each imported file's name; in each, **every annotation name** as one
  region (all its polygons together, as saved now, cleaned or not), and the tissue with all its pieces. Never one
  entry per polygon.
- **Selected polygons**: the polygons picked on the image (click, Ctrl-click to add), read when the action runs.
- In a batch (Selected images), each image uses its own saved annotation of that name. An image without it is
  named and nothing runs, rather than quietly using the whole image.
- In code: `regionOptions(imageToken)` and `regionParams(value, imageToken)` (viewer/histogram.js); the server
  takes `shape=<layer>` with `sid=<id>,<id>` or `name=<name>` or neither (the whole layer), reading layers through
  `annotation_edits.shown`, so ids picked in the browser are found. Segmentation keeps the choice in
  `tab4_mask_layer` and resolves each image's file with `segmentation.saved_annotations`.

### Shared projects

Files SPACEMAP writes are readable and writable by the project's group, not by others (umask 007). Locks open
read-only when they cannot be written. Opening a project repairs the current user's files in it;
`spacemap share-project FOLDER` does it at once. See "Shared projects" in [troubleshooting.md](troubleshooting.md).

## 10. Annotation in detail (1 October 2026)

Tissue has one working annotation. **Detect tissue → This image** uses the current
settings and Label, detects and saves immediately. There is no separate Test or
Original/Clean selector. **Work within stays in both Artifact tabs and Classification**:
it defaults to saved Tissue, allows another loaded annotation and named class, and
follows the image through subsequent steps. Subtract saves into the chosen working
file, with Ctrl+Z and lossless History. External sources use editable project copies;
other classes in the same file are preserved. Legacy Clean Tissue is retained and
adopted or archived according to its source revision. This explicitly supersedes the
older separate Original/Clean workflow and preview-only Tissue detection.

### Classification and interactive heatmaps

(6 October 2026) Classify runs only when pressed: `Classify: This image | Selected`. Settings and Work within never
classify by themselves; opening an image shows its last saved run for that Work within ("saved run"), or says it is not
classified yet. Each run keeps its class map in `outputs/studio/classification/<image-key>/<run>.npz` (with
`<run>.run.json`), so review, Ask AI and Save polygons read it instead of clustering again, and a batch job's results
open at once. The page receives the map once as cluster numbers and colours it itself, so showing or picking a class
needs no request. `Save polygons` takes **Smallest island** (smaller pieces of a class join the nearest class) and
**Smallest hole** (smaller enclosed areas of other or excluded classes are filled), in µm², on the label map before it
becomes polygons; holes of the tissue itself are never filled.

- Use the same Work within and Scale controls as Artifacts. Scale controls the actual
  feature grid and saved outlines; never silently classify at a coarser scale.
- Default to every image channel, including CODEX and H&E. **Selected markers only**
  reads the image layer's checked channels, independent of display thresholds.
  Marker intensity is always used; local texture, edges, Top-K emphasis and smoothing
  are optional feature controls. MiniBatch K-means fits a fixed 20,000-pixel sample.
  Large dense assignments can use CUDA automatically; CPU stays available and never
  waits for CUDA's first import. Viewer colours do not enter the measurement matrix.
- Keep each class in one compact row: visibility, colour, cluster number, adjustable
  top-marker summary, editable name, accept, Ask AI, and optional unmerge/exclude.
  The complete z-score matrix, normalized intensity summaries and pixel counts are
  available to the heatmap and Assistant, regardless of the displayed marker count.
- The left list, map and heatmap share one state. Ctrl/⌘/Shift-click multiple heatmap
  rows to merge. A merge uses combined names until renamed and summarizes all constituent
  pixels (Median by default, optional Mean). Never average class medians. Unmerge restores the original clusters and names before saving.
  Names entered on either side require explicit acceptance; review edits persist as
  project JSON. Save polygons writes the exact reviewed partition, clipped to the
  selected annotation, and keeps earlier outputs in History.
- Shared heatmaps support original order, name order, independent **Cluster X** and
  **Cluster Y** controls, selectable palettes, keyboard selection, tooltips and larger view.
  Keep the labelled colour scale on the right and reserve space for marker and row labels.
  Phenotyping and Classification also offer **Summary: Median | Mean**, with a brief help
  explanation. Use the saved analysis transformations, with no extra arcsinh for corrected
  markers; keep top markers and AI profiles consistent with the displayed summary.
  See [heatmap summaries](studio-heatmap-summaries-2026-10-10.md) for calculations and older runs.
  Apply this convention to Phenotyping, Neighbourhoods and other matrix views too.
  Figure exports use the shown order and corresponding palette. Controls explain
  enrichment units through hover help rather than paragraphs in the workflow.
- Publication heatmaps offer **Current clustering** (the full saved run and its
  transformations) and **Full combined table** (all labelled cells, without sampling).
  Neither follows the open image or selected cells. Use classic's shared renderer,
  independent X/Y dendrograms and right colour scale. As in classic, clusters given
  the same phenotype name are averaged at the profile level for saved-run figures.
  See [publication heatmaps](studio-heatmap-layout-2026-10-10.md) for scope and calculations.
- Ask AI sends the chosen parent annotation name, all class profiles/counts/features,
  selected cluster IDs, a viewer snapshot and exact class map to the configured
  Assistant. Image-capable models receive the images; text-only models receive the
  measurements and the UI reports that distinction. Background suggestions are opt-in.
  Suggestions remain proposals until accepted; merge suggestions require a click.
  Do not treat an enrichment pattern as a diagnosis or a guaranteed cell identity.

Classification review JSON lives in `outputs/studio/classification/<image-key>/`.
It records source revision, channels, feature settings, normalization, class counts,
marker sums, grouping and naming. `annotation_workflow.json` links these records.

### Reusing marker regions and classifications

Marker regions uses the same Work within and Scale rows, settings heading, polygon
list and bottom action row as the other annotation stages. Each compact marker rule
uses raw image intensity; all rules must hold. Scale is the requested detection grid,
with results clipped to the exact selected annotation, including holes. Preview is
optional; Detect regions saves an editable annotation layer immediately.

A result Label names both the saved layer and its initial polygons. A different label
or parent annotation creates a separate result. Rerunning the same label and parent
updates that result with History. The Edit selector chooses which saved result the
polygon list and drawing tools edit. Split, Merge, Name, Delete and Ctrl+Z are the
shared viewer tools; a split retains the parent label until its pieces are renamed.

Generated layers remain the same editable files when selected in Work within. Only
external annotations need an editable import copy. Classification of another layer
or named class creates a separate child layer, leaving its parent and sibling results
intact. For example: threshold Pan-cytokeratin → label Epithelia → Split and name
pieces Tumor/Normal → Classification → Work within that region layer, Class Tumor →
Save polygons. Both the edited parent and subclassification remain available in Layers.

Export lists saved annotation layers with polygon counts. Check the desired layers
and use Export GeoJSON to save and download their combined polygons. Preserve names,
colours, exact geometry, source file/revision and source feature IDs; do not dissolve
overlapping parent and child annotations. Existing exports retain History. The
workflow record (Workflow record) keeps the exported layer revisions. Export never interprets
viewer visibility as inclusion and never exports an unsaved Classification raster.

`outputs/studio/annotation_layers/<image-key>/index.json` links generated files to
their human-readable names, parent selections and generation settings. The workflow
record indexes these derived annotations alongside Classification review records.

### Marker region cutoffs (6 October 2026)

**Cutoff** decides how each rule's level is read, for every image from its own Work within area (the clean tissue
once artifacts are subtracted), so one setting can be tested on one image and run on the selected ones:

- **Raw intensity**: the number typed.
- **Auto level**: the image's background level by the chosen Auto method (the right panel's methods), times a factor.
- **Saved level**: the min shown in the right panel (Auto or by hand), else the Background step's table, times a factor.
- **Arcsinh z-score**: arcsinh(intensity / cofactor), then robust z (median and MAD of the area); default 1.5.

- **Two groups**: the area's pixels (log intensities, smoothed as the rule) split into background and signal by a
  two-group mixture; a pixel counts when it is at least this sure to be signal (default 0.9), and one as bright as
  the signal group's middle always counts. **Fit per** Slide (all ROIs of the slide together, from Samples) or Image.
  The separation of the two groups is reported; below 2 the marker has no clear signal.

Preview, Detect and the job's progress show the raw level each image got ("CD31 above 312 (z 1.5)").
Check the signal on several representative images before running all. Threshold methods can disagree; pooling images is appropriate only when their staining and background are comparable.

### Marker region labels and subtraction

Use **Result label** for the next detection's layer and polygon label. A different
label keeps earlier detected layers; running the same label and scope again keeps
the prior result in History. For existing polygons, check one or more and use
**Apply label** (or Enter). Show the label beside each polygon and on its card.
The text being typed, focus and caret must survive selection and list refreshes.
Labels save automatically; Undo restores previous labels.

Keep optional **Subtract annotations** folded between settings and the polygon
list. Its equation is `[result label] = [source annotations] − [excluded annotations]`.
Each side accepts multiple whole saved layers or individual named classes;
unlabelled regions remain selectable as a whole layer. Include the layer name in
each label's picker context. Preview draws the remainder; **Create layer** saves
an independent editable result and selects it in the polygon list. Reject an
empty result without overwriting anything. Preserve full polygon boundaries and
holes with vector geometry; Scale applies to marker detection, not this subtraction.

Source GeoJSON files remain intact. Store the formula, source paths, class names
and revisions in `outputs/studio/annotation_layers/<image-key>/index.json`, with
operation history linked from `outputs/studio/annotation_workflow.json`. Repeating
the same label/equation updates that result with History; a different label creates
another result. Refresh choices after edits and invalidate changed previews.
Results support the existing split, merge, label, Undo and History tools, and are
available in Work within, Classification and Export after reopening the image.

## 11. Status and open items (5 October 2026)

### Phenotyping input and identity (10 October 2026)

- Raw-marker tables and single-image runs do not require Batch, Slide, tissue or
  corrected-status columns. Leiden, SOM and Vortex retain classic's optional
  filter/grouping behavior: absent grouping means whole-table normalization;
  absent cached row filters are skipped with a job-log message. Present filters
  still apply. New Consensus ML and MR-ML settings leave tissue filtering off;
  explicitly enabled filters and model features remain required and validated.

- Start with **Folders and tables**: Images, Tables, the selected combined CSV/Parquet, Output, then the ROI list with cell counts and per-result completion. Green means every cell has a value in the selected result column; partial coverage is shown separately. Whole-table clustering remains one joint run.
- **Column roles** explicitly maps ROI, physical Slide, unique Cell ID and local Segmentation label. The standard merged table uses `batch_id`, `slide`, `cell_id` and `label`. Never infer that `slide_id` is the physical slide when the user has mapped another column to that role. Link masks by ROI plus local label, checking their positions against the segmentation's own measurements when available.
- Whole-table Leiden/SOM/Vortex and their cluster naming use the selected combined input regardless of the open image's layer table. An image does not need to be open. Preserve global cell IDs; reject missing or repeated IDs before launching a job.
- Batch correction's **Use this table** saves the same input for Studio, project reopening and classic Phenotyping. Offer direct Parquet alongside CSV; cache companions are not extra datasets.
- Label the transformation setting **Normalise within**. Include the actual column name in each option (for example, `Each slide (slide)` versus `Each value of slide_id`). This controls marker normalization, not separate clustering runs. Publication figure controls start folded.
- Before Save/Run, read the visible transformation controls, keep one live draft object across saves, and check the acknowledged settings revision before starting. A previous result's heatmap continues to describe its own saved run; changed transformations require a new run.
- Every automated Studio phenotyping method has **Overwrite existing results** beside its output controls, off by default. A collision without it stops the run rather than silently adding a suffix. With it, the selected outputs and associated clusters/scores are replaced together on success. Re-clustering resets old phenotype names. Repeated **Save names** updates the chosen run's phenotype column; earlier name mappings remain in its jobs, not extra `_2`/`_3` columns. See [settings and overwrite](studio-phenotyping-settings-2026-10-10.md).
- **View data** opens the selected combined table in the bottom panel, including newly saved result columns and numeric scores. Keep paging on the server. Cluster list and heatmap selections agree with image visibility and Data's **Selected cells** filter; cell clicks and Data rows select their clusters in return. Always include ROI in local-cell identity, including whole-table navigation.
- **Ask AI** on clusters uses the selected run, comparison profiles and reference labels, then fills the name fields and heatmap labels. Suggestions remain editable; explicit naming/apply instructions authorize the registered save action without a second confirmation. Conversational approval applies the stored proposal. Show saving/completed/failed from command and job receipts, and switch cell colouring only after the save succeeds. Changing or clearing a subset preserves the other names, with earlier mappings retained in job requests. Keep explanations in help controls. See [linked review](phenotyping-linked-review-2026-10-10.md) and [active naming](phenotyping-assistant-actions-2026-10-10.md).
- **Name suggestions** is one panel containing **Model**, **Suggest names** and **Ask AI**.
  Leiden, SOM and Vortex offer compatible trained Manual reference models and
  existing reference columns. The saved model works without AI or labels in the
  target table. It samples up to 400 cells per cluster using its own features and
  preprocessing; marker display levels and heatmap transforms do not alter it.
  Model names match Manual reference. Use only matching features available in
  this clustering, including when it has fewer features than the training panel.
  Adapt from saved training examples and check saved validation on that subset;
  retain the original model. Match marker identity across raw/corrected names
  while reading the model's original measurement columns and preprocessing.
  Model loading starts only on request, and all cluster samples share one
  measurement read. A later source selection wins over a delayed result.
  Show one clickable suggestion per cluster; alternatives, manual counts,
  sample coverage, missing features and saved validation belong in `?` help.
  Predictions and manual counts remain separate, and model scores are not
  independent naming accuracy. **Ask AI** receives the selected model evidence
  with the heatmap and numerical profiles. Suggestions preserve typed names;
  click one to fill a name, edit if needed, then **Save names**.

The live status, what is done (with commits) and the next checklist are kept in one place:
"Status, 5 October 2026" in [studio-checklist.md](studio-checklist.md). This section only summarises.

- Studio steps in this design: Project (6 October), Registration, Background, Regions of interest, Annotation. Next: Segmentation, then
  Cell tagging.
- The other model's Annotation work (29 September to 1 October) is committed and its problems fixed.
- New: the model library (Project step) and group-shared project files.
- Open: shared analysis code to bring to the main checkout before merging; a failed Annotation settings save no
  longer says so; "Start live RESOLVE" is folded inside Layers; phenotyping models trained on another scanner may
  need their markers put on the same scale.

### Annotations in the later steps (2026-10-09)

- **Segmentation**: Selected returns at once: each image's area is cut out of its annotations inside the job (an image
  whose area cannot be made is skipped and said). Files Studio finds next to the images are checked for being exports
  from their first 256 KB only, the answer kept between restarts (the first Segmentation load had read 249 files whole).
  Each image's Area list has two groups, Local and Imported, by name (`local:tissue`,
  `imported:tissue`): the same values for every image, so Area (set all) → Imported › tissue segments each image inside
  its own imported tissue. A choice saved by plain name (tissue) shows as Local › tissue; one saved by file
  (`<file> › <name>`) is listed as Earlier choice. Exported files are never listed.
- **Cell tagging**: **Annotations from** (all tables): the Annotation step's export (else the folder's file by name),
  the GeoJSON imported for each image (the newest), or the Annotations folder by name; a table can still pick its own
  file, its imported ones listed first. Typing an Annotations folder there keeps it (it no longer snaps back to the
  Annotation step's shared folder).
- **Background** works on the full images (before ROIs) and keeps its Annotations folder matched by name (now also by
  words and numbers); **Class (set all)** sets the class for the ticked images that have it.
- **Cells in Layers**: the image's cells are the Segmentation step's own (its Output's geojson/ and cell table) when it
  has segmented the image; the image on screen reopens with them when its segmentation finishes while Segmentation is
  open. Cell outlines dropped together with a table of the same name (P11_roi_1.ome_cell_boundaries.geojson and
  P11_roi_1.ome_cell_features.csv) make one cells layer whose data is that table; an imported segmentation otherwise
  takes the table of the same name beside it (a run's <image>_cell_features.csv next to its geojson/).

### Lists: project files across steps (23 October 2026)

The right dock order is **Layers, Lists, Jobs, Assistant**. Lists replaces the
step-only Outputs tab. Its tooltip names inputs, outputs and imports. Each Studio
step has a collapsible section using the same name as the rail; file groups are
collapsed beneath it. Selection, search, Import, Export and Remove share the group
header and disappear when folded. Dense rows retain the existing font and control
sizes. Long lists page their DOM rows; select-all covers every search match and
keeps selections outside the search. Last-action messages belong with Jobs.

Registration lists CODEX and H&E inputs and registered images. Background lists
threshold tables and background-removed images; Regions of interest lists ROI
outlines and extracted images. Downstream inputs already represented by an
upstream output are not repeated. Annotation separates imported GeoJSON, tissue,
each artifact kind, classification, marker regions, zones and exported GeoJSON.
Every imported copy has its own stable file identity, including duplicates and
copies whose original image folder is no longer configured. Class initials have
full-name/count tooltips. Segmentation lists cell measurements, outlines and
models; Cell tagging lists its resulting tables. Batch correction lists combined
tables. Phenotyping groups its methods and places models within Manual reference.
Spatial analysis and Visualization list saved results and figures.

File hover gives the absolute path; right-click offers Copy path. Opening a row
selects its workflow, opens the matching image and table or annotation, and shows
Layers. A class badge reveals that class in the exact source. Imports accept file
selection, paths/folders and drops into a compatible group. Model archives carry
weights and metadata; listing them never deserializes model weights. Remove in Lists only unlists the item from this project. All files and sidecars
stay unchanged at their existing paths, including outputs, uploaded copies and
models. Physical cleanup is a separate explicit operation.
Annotation imports leave Layers and segmentation area choices immediately.
Selection of a deleted area remains invalid rather than falling back to the whole
image. Inventory scans and table/class summaries are requested lazily, separately
from startup and image opening.

GeoJSON path/folder imports register original paths in a small project index;
they do not copy polygon data. Linked files remain read-only until Edit a copy.
Uploads retain a project file because the browser cannot provide a usable source
path. Folder imports include compatible direct files only, ignore subdirectories,
and match independently of the open image, retain unmatched files with Match, and report failures without
stopping the rest of the batch. Repeated references remain distinct in Lists;
removing a reference never deletes its original. Artifact Layers contain polygon
review/editing only; prediction parameters remain in the left Annotation panel.

Jobs → Storage → Temporary files scans only on request. Expand a category to
review individual files, sizes and full paths. Hover describes purpose and impact;
right-click offers Copy path. Selection can cover individual files or a category.
Cleanup checks output columns, saved model versions or the exact saved annotation
version, revalidates before deletion, and preserves both Mean and Median cluster
profiles before removing a Leiden working table. It keeps unknown/unique results,
pending label files and active jobs. Annotation-history removal is optional and
explicit. Clear finished continues to hide job cards; it does not imply disk cleanup.

Jobs → Storage → Performance caches separately inventories viewer outline pyramids,
per-image tables, Flow plots and memberships, Leiden reload files, verified CSV
fast copies, shared cell indexes and summaries, compiled calculations and downloaded
model resources. Components and their files have sizes, full paths, impact hover
hints and Copy path. These remain outside temporary cleanup selection. Flow's
existing explicit clear action stays within its expanded cache group. Cache scans
wait for expansion; project changes invalidate previews and outstanding responses.

Studio's Steps, Parameters and right dock can each collapse independently from
small circular arrows beside their headings: left arrows for Steps/Parameters,
and a right arrow immediately before Layers. Hidden panels get restore arrows
in the top header. Each width and its controls survive folding; navigating
to a right-dock tab reveals it. Marker tiles and Data each float into a separate
same-origin browser window. Live DOM controls move between windows so table edits,
filters and linked image selection remain shared. Dock back or closing a child
window returns that view independently. A blocked popup leaves its view docked.

Cell tagging offers **Update existing table** beside the merged CSV/Parquet choice.
The current image or selected images and ticked annotation fields define the update.
**None** clears tissue, annotation, bed and edge column selections so users can
choose just one correction. Unticked fields and unselected rows remain unchanged;
new fields are blank for unprocessed images. Existing matching CSV and Parquet
copies are staged together, checked for matching cell identities, and replaced
with retained backups under `.cell_tagging_backups`. Pending saved edits, changed
inputs, and conflicting recorded units block publication. This does not rerun
batch correction or phenotyping; analyses depending on changed fields need review.
Completion refreshes the unified Cell data through the existing result-refresh path.

Cell tagging progress redraws its status/list without reapplying the cell preview.
Finished jobs attach the current image's published tagged table to its existing
cell layer, including when another workflow is open. Other images do not take
over the viewer. Zoom, image tiles, markers and cell filters stay in place; Data
following those cells retains its filters and sorting. Matching preview columns
switch to their saved values. Current batches publish tables together at the end;
staging files are never loaded. Identity checks reject changed segmentations, and
delayed responses cannot override later image/source/colour choices or Data edits.

The same rule applies to phenotyping and other jobs that save cell columns:
progress only updates status/lists. Each completed result refreshes linked data
even if other jobs are running or another step is open. The saved-table watcher
and job completion compare the same file/label revision, so unchanged results
do not invalidate cell tiles twice. Background updates retain the visible overlay
until replacement tiles have loaded. Image channels, viewport, cell-data choice,
colours, hidden groups and Data filters remain in place.

Layers groups annotations into Working annotations and Imported GeoJSON. Working
contains Tissue and Artifacts, with artifact categories containing independently
collapsible Bad tissue and Good tissue lists. Closed roles do not build polygon
editors. Different imported files retain separate identities; hovering their
names shows absolute paths. The same saved file is displayed once when it already
has a working polygon layer. Explicitly imported sources retain their provenance
even when their filenames resemble generated outputs.
