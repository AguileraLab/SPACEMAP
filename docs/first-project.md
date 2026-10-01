# Your first SPACEMAP project

Complete [installation](installation.md), run `spacemap doctor`, start
`spacemap` and open <http://localhost:8501>. You can create and reload an empty
project before downloading study data.

## Create the workspace

In **Tab 0 — Project Workspace**, find **Configure New Project**:

1. Enter a **Project Name**, such as `MyStudy`.
2. Choose an existing **Parent Experiments Directory** outside the software clone.
   Create that parent folder first if it does not exist.
3. Click **Create and Initialize Project**.
4. Check the displayed root and directory settings. Normally the app creates
   `project/MyStudy` beneath your chosen parent; a parent already named
   `project` is used directly.
5. Save after changing directories. Keep `<project_name>.json` with the inputs
   and outputs: it records the project's configuration.

Reopen through **Load Existing Project** or **Browse Project Root Directory**.
Select the folder holding `<project_name>.json` (or an older `experiment.json`).
Review directory mappings when moving a project between computers.

## Prepare inputs

Start with a small image or ROI to check alignment, channels and segmentation.

| Input | Used for | Check before running |
| --- | --- | --- |
| Multiplex TIFF/OME-TIFF or supported slide images | Image processing | Channel names, nuclear channel, axis order and pixel size |
| Matching H&E images | CODEX–H&E registration and H&E annotation | Correct pairing and H&E pixel size |
| GeoJSON ROI boundaries | ROI extraction | Coordinates match the selected image |
| Reviewed tissue annotations | Annotation and cell assignment | Class names and alignment with ROI images |
| Cell feature CSVs, if images are already processed | Batch correction onward | Stable cell/sample IDs, marker names and cell coordinates |

Default inputs include `data/codex`, `data/he`, `data/boundaries` and
`data/annotations`. Tab 0 can map existing folders instead. A new project does
not contain example images or pretrained custom model weights.

Enter the correct **micrometres per pixel (MPP)**. The manuscript analyses used
**0.5083 µm/pixel for INNATE** and **0.3775 µm/pixel for the Schürch/Nolan
images**. These are study-specific calibrations. Other images, including the
matching H&E, may use different values.

## Follow the stages

Each page's **Step Guide** lists inputs, outputs and checks. Start where your
available inputs enter the workflow; use directory mappings to connect
previously processed images or tables.

| Tab | Action | Expected output |
| --- | --- | --- |
| 1 — Registration | Match images, verify both calibrations and inspect alignment | Registered image stacks |
| 2 — Background subtraction | Detect channels and check background assignments | Corrected images |
| 3 — ROI extraction | Load/draw boundaries and select channels | ROI images |
| 4 — Annotation | Identify tissue, review in QuPath, import/export reviewed classes | Tissue GeoJSON and zones |
| 5 — Segmentation | Choose the nuclear channel, model and tissue mask; inspect a small result | Cell masks, boundaries and feature tables |
| 6 — Batch correction | Scan cell CSVs, select markers, filters and correction method | Combined/corrected cell table |
| 7 — Phenotyping | Check the table and marker panel; run the chosen methods | Phenotype assignments and QC plots |
| 8 — Spatial analysis | Confirm coordinates, units and phenotype column | Spatial measurements and tables |
| 9 — Visualization | Select the table, images and overlays | Exported figures |

Cellpose is included; initial weights may need a download. DeepCell/Mesmer
needs separate installation and model setup. The shipped marker panel is a
starting example: check it against your antibody panel before analysis.

Outputs normally go under `outputs/`. Check previews, counts and the Process
Log before proceeding. A page opening successfully does not validate image
calibration or biological results.

## Save and share

Retain the project JSON, input manifest, software commit/version, environment
versions and outputs. When a phenotyping page offers **Write Labels Into CSV**,
use it before sharing: pending labels may otherwise exist only in companion
files and be absent from the CSV opened by another program.

This repository supplies software. INNATE images, the unified study CSV and
custom model weights belong in the accompanying reviewer/public data package.
Use that package's README and saved parameters to reproduce the manuscript;
a general installation guide does not replace the study package.

## Common first-run issues

- **No images:** check the selected stage's input folder on the computer/server
  running SPACEMAP.
- **Misaligned overlays:** check image/boundary pairing, coordinates, channels
  and MPP.
- **No GPU:** CPU operation is supported but slower. Check driver/PyTorch setup
  if you expected acceleration.
- **Port in use:** run `spacemap --port 8502` and open that port instead.
- **Dependency error:** run `spacemap doctor` and `python -m pip check` in the
  environment used to start the application.

See [Troubleshooting](troubleshooting.md). Reports should include the SPACEMAP
version, operating system, installation route and full error text.
