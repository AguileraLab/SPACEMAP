# Image viewer

SPACEMAP has a built-in image viewer. It opens raw scans, registered images
and ROIs at full resolution, with the results of every tab drawn on top: cell
outlines, phenotypes, neighbourhoods, tissue annotations and any other
per-cell column. You can also use it to label cells by hand (Manual
Reference).

It runs in the browser tab SPACEMAP already uses. There is nothing extra to
install or forward.

## SPACEMAP Studio

**SPACEMAP Studio** opens the same viewer with analysis steps on the left,
layers on the right and an Ask bar at the top. Classic SPACEMAP remains
available. Work lists show saved outputs and recorded problems; other steps
link directly to their classic tools. The image stays open when switching
steps or using view commands.

In Phenotyping, Live RESOLVE shares its saved options with classic Tab 7.
Edit the panel, save raw marker thresholds and run all ROIs as a background
job. Jobs show progress, Stop, a log and **Show result**. Results use new
columns and the existing pending-column store.

**Select and label cells** offers marker rules, click/box/polygon refinement,
an exact marker comparison, assignment into a reference column and a saved rule
library. Selection checks every cell. Display histograms use clearly labelled samples; the heatmap uses every cell. Z1 compares cells within an ROI; Z2 compares markers
within a cell; Arcsinh uses cofactor 5. Drawn selection regions use cell centres.

**Train from reference labels** compares two saved suggestion models. Choose
a reference column or explicitly checked cells, then choose the saved model in
the manual-reference controls. Training runs once in the background. Scores
measure agreement with held-out reference labels, not biological certainty.
MR-ML uses classic SPACEMAP's implementation and saves new prediction columns.
**Only cells marked checked** applies to **Train suggestion models**: enable it
to use only cells approved with **Mark checked** for the chosen reference column.
When disabled, other labelled cells can also train the model. **Needs review**
cells are excluded either way. It does not restrict MR-ML's prediction targets.

**MR-ML from one column.** Choose **Reference column only** and a column such
as `phenotype_ref1`. Valid reference labels are retained; blank, Unassigned,
Unknown, Unclassified and explicitly excluded labels are predicted from measured
markers. **Also predict** lists additional labels to exclude from training
(initially `Others`). At least two reference classes are needed, with enough
examples for a stratified validation split. **Morphology and other measurements**
adds numeric features such as area and eccentricity. These are optional, used
alongside the marker measurements, and must not be phenotype labels or cell IDs.
Cells with incomplete selected measurements remain unpredicted; known reference
labels are retained even when measurements are missing.

Name the **Predictions**, **Score** and **Filtered labels** columns before running.
Scores range from 0 to 1 and are model support, not calibrated probabilities.
Retained reference labels have blank prediction scores. In the filtered column,
predicted cells below **Minimum score** receive **Below cutoff** (initially
`Others`). **Show result** displays this filtered column. Results are saved in
small Parquet column files, read immediately by both interfaces; **Jobs → Saved
results → Write into the CSV/table** incorporates them into the source table.
For a loaded Parquet table, **Save a CSV copy** (or **Update CSV copy**) exports
the saved columns and label edits. Classic MR-ML also accepts an empty optional
method list and offers morphology selection, named outputs, **Save All
Assignments to Disk**, and the shared pending-column CSV write controls.
**Combine additional methods** retains the ensemble workflow.

**Cells and maps → Top markers** mirrors the annotation control. Turn it on and
set **Number** to show only the clicked cell's highest measured markers. The
ranking matches the cell's Markers panel, including the selected Raw, Arcsinh,
Z1, Z2, CDF and NLog transformations. Changing transformations updates the shown
channels. Colours and contrast limits remain as set; turning the option off
leaves the current channels unchanged.

In Inspect mode, **Ctrl-click** (Command-click on macOS) adds or removes cells;
a normal click returns to one cell. Selected outlines are highlighted together.
The card shows label counts for the entire selection, arithmetic mean marker
values after transforming each cell, and suggestions based on mean individual
model scores across all phenotypes. Prediction coverage is shown when some
cells lack usable measurements. **Assign all** explicitly applies a suggested
label to the whole selection. Top markers follows the selection's average
ranking. Late responses cannot replace a newer selection or change channels
after the option is disabled.

Ask can open an image, show markers, colour/filter cells, show density, count,
select cells and compare markers. Common wording works without AI. Other
wording uses the configured local model and the same checked commands.
Changes proposed by chat require confirmation. **Describe this view** sends a
small preview to the configured local image model and labels its response as
an AI description. Undo is offered for actions that can be reversed safely.

See the [Studio test list](codex/studio-test-list.md) and
[selection/training details](codex/studio-stage-3-2026-09-26.md).

## Opening it

- **From any tab:** press **Open image viewer** at the top of the page, next
  to **Record run log**. Pick an image from the list. The list holds every
  image the project has:
  - raw CODEX and H&E scans (`.qptiff`);
  - registered images;
  - background-subtracted images;
  - ROIs.
- **From Tab 3 (ROI Extraction):** **Draw ROIs in the image viewer**, under
  ROI Definition, opens the image with its ROI file as an editable **ROIs**
  layer (see [Drawing shapes](#drawing-shapes)).
- **From Tab 1 (Registration):** each sample pair has three buttons:
  - **View CODEX** opens the raw CODEX scan with the markers and colours
    chosen on that page.
  - **View H&E** opens the raw H&E scan.
  - **View registered result** opens the merged image with the registered
    H&E laid over it at half opacity. Use this to check the alignment.

**Working beside SPACEMAP.**
- **Separate window** (toolbar) opens the viewer in a browser window of its own. Move it, resize it or put it on a second screen, and keep working in SPACEMAP. While it is open, images opened from any tab show in that window instead of the pop-up, and cell colours update when a tab writes new columns or labels.
- The window has its own **Open another image...** list, so you can move between the project's images without going back to SPACEMAP.
- **Full screen** fills the screen with the viewer. It works in the separate window; the browser does not allow it inside the pop-up.
- **Minimize** closes the pop-up. **Restore image viewer**, under **Open image viewer**, brings it back as you left it.
- The view (position, zoom, layer settings and order, added layers) is remembered for each image, in this browser.

The viewer opens on what the current tab produces:

| Opened from | Image list starts at | Cells coloured by |
|---|---|---|
| Tab 1 | Registered images | (outlines) |
| Tab 2 | Background-subtracted images | (outlines) |
| Tab 4 | ROIs, first annotation layer shown | (outlines) |
| Tab 0, 3, 5, 6 | ROIs | (outlines) |
| Tab 7, 9 | ROIs | the newest `phenotype_*` column |
| Tab 8 | ROIs | the neighbourhood column (`neighborhood`, `CN_name_merged`, ...) |

You can change any of these in the viewer.

## Using it

**Moving around.** Scroll to zoom, drag to pan. **Fit** shows the whole
image. **100%** shows one source pixel per viewer pixel, independent of the
image's µm calibration. Right-click for percentage zoom presets. Enlargement
cannot add source detail.
The bottom-left corner shows the scale bar and the position under the mouse,
in µm when the image records its pixel size.

Native-resolution image tiles and full-resolution figure exports use lossless
PNG after the selected channel display settings are applied. Overview tiles
use compact JPEG without colour subsampling. The viewer selects pyramid
detail for the zoom and display density. Enlarged pixels are crisp by default;
right-click and enable **Smooth enlarged pixels** if you prefer interpolation.
This preference is remembered per image in project JSON. Reductions remain
filtered to avoid aliasing. These display choices do not alter source images
or the pixels used for analysis.

**Marker tiles.** Open the bottom **Marker tiles** handle for individual marker
crops at the cursor. **Selected markers** follows the image checkboxes, names,
colours and contrast limits; **All markers** shows the other channels without
changing the main image. Cell and annotation Top markers therefore update the
tiles as well. Marker transformations rank measured cell features; the tiles
continue to show image pixels with their display ranges.

Clicking a cell pins the panel to it; Ctrl-click fits the selected cells into
the same crop. **Follow cursor** releases the pin. Use the zoom menu or
right-click for percentages (100% is native image pixels), **Size** for card
size, and drag the top edge for panel height. Scroll to reach more markers.
**Float** puts the panel above Studio's sidebars: drag its header to move it,
or its bottom-right corner to resize. **Dock** returns it to the viewer bottom;
**Hide** collapses it to the handle. Display preferences are saved with each
image's project JSON; transient cursor positions and cell selections are not.
Only visible cards load, using cached lossless tiles and bounded requests.

**Ask AI about a cell.** The cell suggestion card's **Ask AI** sends the current
measurements (including the selected transformations), numeric morphology and
other features, current phenotype calls, model scores when available, and
phenotype names from the model/dataset to the configured Studio Assistant.
It attaches labelled contact sheets of the selected cell's marker crops, using
the Marker tiles scope and zoom, even when the drawer is hidden. White outlines
identify the selected cells. For a multi-cell selection the measured profile is
the mean after transforming each cell; a mixed population may be ambiguous.
The reply is advisory and never applies labels. If the configured model cannot
read images, Studio says so and requests a measurement-based review instead.
Changing the inspected cells or measurement settings cancels an unfinished
capture so a request cannot mix old images with new measurements.

**Layers.** The panel on the right lists the layers. Tick a layer to show
it and click its name to open its settings. Every layer has an **Opacity**
slider.

- **Image.** Tick markers to show them. Each marker has a colour and a
  display range: values at or below the first number are black, values at or
  above the second are full colour. **Auto** picks the range from the image.
  Type in **Find a marker...** to filter a long list. Marker names come from
  the file itself (OME channel names, or the Biomarker names a PhenoCycler
  scan records). Your choices are saved with the project, in
  `outputs/viewer/display/<image>.json`, and come back the next time the
  image is opened, on any computer.
  - **Marker sets.** Tick the markers you want together (for example CD3e,
    CD4 and CD8), set their colours, ranges and names, and press **Save
    ticked as set...** to name them ("T cells"). Choosing the set later shows
    exactly those markers, as you saved them, and hides the others. Sets are
    kept for the whole project (`outputs/viewer/display/marker_sets.json`)
    and work on every image that has those markers.
  - **Histogram.** Click a marker's name to see its histogram under the
    list, as in QuPath's Brightness & Contrast. Drag the two lines to set
    the range: pixels below the left line are black, so moving it up hides
    the background and shows what a background threshold at that value
    would leave. **Measure in** switches the histogram from the whole image
    to one drawn shape, for example an area of background. The **Name**
    box renames the marker; Tab 3 can take the new names and colours into
    the extracted ROIs (**Use names and colours from the image viewer**).
- **H&E (registered).** On a merged image that holds the registered H&E, this
  layer shows the H&E in true colour.
- **Cells.** Cell outlines from Tab 5.
  - **Draw** chooses outlines, filled cells or both.
  - **Table** chooses the per-cell table: the unified table, or a
    cell-level table from Tab 8.
  - **Colour by** takes any column. Text columns such as phenotypes get the
    same colours as in Tab 9. Numeric columns (marker intensities, distances,
    areas) use a colour map.
  - The legend lists each value with its cell count. Click a value to hide
    those cells.
- **Annotations.** Tissue annotations from Tab 4 (and QuPath GeoJSON in the
  annotation folders), filled in their class colours. Click a class in the
  legend to hide it.

- **Drawings.** Shapes you draw on this image (see below).

**Layer order.** The top of the layer list is drawn on top. Move layers with
the arrows beside their names; drawn shapes always stay on top.

**Inspecting a cell.** With the tool set to **Inspect**, click a cell. A
card shows its labels (phenotypes, neighbourhood, ...) and its top markers,
ranked by one of:
- **Z1** (default): how unusually high the cell is for each marker, compared
  with the other cells of this ROI. It is RESOLVE's Z1 with batches grouped
  by ROI: `arcsinh(raw / 5)`, then a z-score per marker.
- **Z2**: which markers stand out within the cell. This is the cell's Z1
  values as a z-score across its own markers (RESOLVE's Z2).
- **Arcsinh** or **Raw** values.

The ROI's marker values are read once, in the background, when the viewer
opens, so each click takes about a millisecond.

**Density maps.** In **Add to the view**, **Density map** adds a smoothed map
of cells, made as Tab 9's virtual tissue: centres binned, then a Gaussian of
the chosen radius (µm).
- **Cells:** all cells, or tick the groups of any column (for example CD8 in
  `phenotype_leiden`). The map takes the group's colour.
- **One map per group** turns several ticked groups into one map each, in
  their colours. Overlaid, they show where each group is, and where they
  meet or overlap. Use the arrows to choose which is on top.
- **Show:** cell density (cells/mm²), or the local average of a column (a
  marker, a distance), optionally faded where cells are sparse.
- **Colours:** one colour (clear to solid, for overlaying) or a colour map.
- **Triads:** once Tab 8 has found triads, their instance files
  (`outputs/spatial_analysis/triad/triad_instances*.csv`) appear in **Table**.
  Choose one to map triad density, all or by triad type.

**Other GeoJSON files.** **GeoJSON file...** shows a file from your computer
(QuPath or other tools, in the image's pixel coordinates); it is kept in
`outputs/viewer/imported/`. **Project GeoJSON...** shows any GeoJSON the
project already has. Both are drawn by the server, so large files are fine.

**Saving an image.**
- **Save image** saves what is on screen as a PNG, with the scale bar.
- **Save full resolution** saves the same area at full image resolution, with
  every visible layer at its opacity and a scale bar. Use it for figures. The
  longest side is at most 8,192 px; a larger area (a whole ROI, say) is saved
  at half or quarter resolution, and the message and file name say which.

## Drawing shapes

Layers of the **Shapes** kind hold regions you draw: ROIs in Tab 3, and a
**Drawings** layer on every image opened from the image list. When one is
shown, the tools appear above the image:

- **Move:** pan and zoom; click a shape to select it.
- **Rectangle:** drag.
- **Polygon:** click the corners; double-click, press Enter or click the
  first corner to finish. Backspace removes the last corner.
- **Freehand:** drag around an area.

Esc cancels a shape being drawn. Delete removes the selected shape.

The layer's card lists every shape with its area. There you can rename a
shape, change its colour (or its class, where the tab offers classes such as
"ROI"), zoom to it (**Go**) and delete it. **Undo** reverses the last change.
**Import from the project...** adds the shapes of a GeoJSON file the project
already has (annotations, other ROI files); **Import a file** adds a GeoJSON
from your computer, such as a QuPath export in the image's pixel
coordinates; **Download** saves the layer as GeoJSON.

Shapes are saved as you draw, as GeoJSON in full-resolution pixels. Each
shape keeps `name` and `label` (the same text) and a QuPath-style
`classification`, so the file works in SPACEMAP, QuPath and other tools. Very
detailed shapes (tissue outlines with hundreds of thousands of corners) are
simplified only for drawing on screen; the file keeps every corner.

## Labelling cells (Manual Reference)

The **Label cells (Manual Reference)** card appears when an ROI has cells.

1. **Column:** where the names go. The default is `phenotype_ref1`. Pick an
   existing reference column from the list or type a new name.
2. **Name:** what to call the cells, for example `CD8 T cell`. Names already
   in the column are suggested.
3. **Tool:**
   - **Click cells:** click a cell to give it the name.
   - **Draw box:** drag a box to name every cell inside it.
   - Hold **Shift** with either tool to clear the name instead.
4. **Undo** reverses the last step. **Show this column** colours the cells by
   the column you are filling, so you can see your progress.

Each label is saved when you click. Labels are stored beside the table, as
the Tab 7 Manual Reference tool stores them, and every tab sees them at once.
To put them into the CSV itself, use **Write pending changes into the CSV**
in the file panel at the top of Tab 7, 8 or 9. Do this before you use the
CSV outside SPACEMAP.

## Starting SPACEMAP

`./run_interface.sh` and the `spacemap` command start
`src/spacemap_gui/server.py`. It runs the same interface as `app.py` and also
serves the viewer's image data on the same port. To run a second copy beside
another one, give it its own port:

```bash
./run_interface.sh --server.port 8503
```

If SPACEMAP is started with `streamlit run src/spacemap_gui/app.py`, the viewer
still works: it serves its image data on a second, free port. On a cluster,
that second port must be forwarded as well, so prefer `server.py`.

## Speed

- Images are read tile by tile from the pyramid levels already stored in the
  files, and only the part on screen is read. A whole-slide PhenoCycler scan
  opens in about 0.1 s; a tile takes 20-90 ms the first time and a few ms
  after that.
- The first time an ROI's cells are shown, their outlines are turned into a
  label image (a few seconds and about 8 MB per ROI). This happens in the
  background: the image shows at once and the cells appear when ready. It is stored in
  `outputs/viewer_cache/cell_labels/` and rebuilt only when the segmentation
  file changes. The folder can be deleted at any time.
- Colouring by a column reads just that column for the ROI. When the table
  has a fast-loading copy (Parquet), this takes a fraction of a second.

## For developers

```
browser (OpenSeadragon, frontend/viewer/*.js)
   |  GET /spacemap-viewer/api/...   (same origin as the interface)
   v
spacemap_gui/viewer/api.py      Starlette routes, mounted by server.py through st.App
   |  tokens -> files            spacemap_gui/viewer/registry.py
   v
spacemap/viewer/sources.py       tifffile + zarr: QPTIFF, OME-TIFF (per-channel pages + SubIFDs), RGB
spacemap/viewer/render.py        tiles: level choice, per-channel colour and range, additive blend, JPEG
spacemap/viewer/cells.py         label pyramid, colour look-up per cell, labelling through spacemap.labels
spacemap/viewer/annotations.py   GeoJSON -> STRtree -> rasterised per tile
spacemap/viewer/catalog.py       which images, segmentations, annotations and tables a project has
spacemap/viewer/shapes.py        drawn shapes: GeoJSON store, display simplification, masks
spacemap/viewer/display.py       saved marker settings; minimums() and renamed() for other steps
```

- **Tile addressing.** `z` is the power-of-two downsample (0 = full
  resolution) and tiles are 512 px. The tile source in `viewer/setup.js` maps its
  level `L` to `z = maxLevel - L`. Each tile is read from the stored level
  closest in size, then resized with area averaging.
- **Restarts.** Tokens are signed digests of what they stand for, with a
  key kept in the user's SPACEMAP state folder, and the token table and each
  window's last view are saved there (`viewer_registry.json`). Links, open
  windows and remembered layers keep working after a restart.
- **Security.** The browser never sends a file path. `registry.register`
  accepts only existing image files and returns a random token; the routes
  serve tokens only. Cell and annotation layers are registered the same way,
  with `register_object`.
- **Caches.** Raw tiles are kept in a 512 MB in-memory cache, and label
  tiles, colour look-ups and ROI indexes in smaller ones. Each cache is keyed
  by the file's size and modification time, so a changed file is never shown
  stale.
- **Streamlit component.** `frontend/` is a plain HTML and JavaScript
  component with no build step. OpenSeadragon 6.1.1 is vendored under
  `frontend/vendor/` with its BSD-3 licence.
- **Adding a tab's own button.** Build layers with `ui.image_layer`,
  `ui.he_overlay`, `ui.shapes_layer` or `ui.layers_for_image`, then call
  `ui.open_viewer(title, layers)`. Tab 1 (`_render_pair_viewer_buttons`) and
  Tab 3 (`_render_roi_viewer_button`) show how. To change what a tab opens
  on, edit `catalog.start_group` and `catalog.suggest_colouring`.
- **Getting results back into a tab.** The viewer never reruns the page
  while you work in it. What it produces is saved into project files through
  its routes, and the tab reads them when you press one of its own buttons:
  shapes with `spacemap.viewer.shapes.load(path)` (or `shapes.mask(...)`),
  marker settings with `spacemap.viewer.display.minimums(path)` and
  `renamed(path)`, labels through the Manual Reference label store.
- **Tests.** `tests/test_viewer.py` builds small synthetic files (a QPTIFF
  laid out like the scanner writes it, a SPACEMAP OME-TIFF, GeoJSON outlines
  and annotations, a unified table). It checks reading, tiles, colouring,
  labelling with undo, the routes and the pop-up.
