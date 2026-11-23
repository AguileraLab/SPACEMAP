# Cell data and cell inspection

The **Cell data** selector at the top of **Layers → Cells** is shared
by cell colouring, filters, density maps and the Data tab. The Data tab displays
the current choice and has a **Change cell data** shortcut to the source list.

Cells uses the outlines and data already loaded in the viewer. Importing or
dropping a CSV/Parquet or segmentation adds an available source without changing
the current table, colours, filters or edits. The bottom **Add / switch cells…**
list separates **Loaded outlines**, **Linked cell data**, and **Imported tables**.
Choosing an imported table opens it in Data only. **Link to Cells…** lets the user
choose matching columns and link it to the current outlines; linking does not
activate it. Choose the linked table explicitly to use it for colouring and maps.
Imported tables retain their own filenames and identities. Unsaved Data edits
must be saved or discarded before switching tables.

The expanded Cells header contains visibility and **Top markers**. Marker-set
and scale controls appear only when Top markers is on. Appearance and opacity
share a row; Filters precede Colour by. Categorical colour keys have searchable
checkboxes and editable swatches. Numeric columns have a palette and range.
**Sources…** retains per-source visibility, export and geometry settings.

- **Segmentation measurements**: the original per-cell measurements associated
  with those cell outlines.
- **Cell tagging results**: measurements with labels added in Cell tagging.
- **Phenotyping results**: the shared measurements, phenotypes and clusters used
  by Phenotyping and Flow gating. Only the current image's cells are drawn.

The **?** beside Cell data holds its description, filename, and any incompatible
file details. The normal panel shows only the selector and **Open data**.
A missing source shows a placeholder and disables Open data; the first menu
option must not appear selected by accident.

**Cells shown** identifies the current selection. A new cluster click, Flow
selection or Data filter replaces the previous selection. Delayed responses
cannot restore an older choice. **Show all cells** resets the view's cell filters.
Filtering cell overlays leaves the underlying marker image visible.
Changing Cell data clears the old source's selection and closes its inspection
card; the next cell click reads the newly chosen data.

**Density** beside Colour by creates a map from the current cell source and
selection. Categories default to counts per area; numeric columns default to
their local mean. Map value and Colours share a row, as do Opacity and Radius.
Under **More**, **Cells included** offers **All cells in this image**, **Follow
shown cells**, or **Keep this selection**. Following uses the current filters,
numeric ranges and Flow gates. Keeping a selection pins its table and selection
criteria; it does not copy or freeze the underlying data values. Optional map
filters can narrow it further.
Legacy point/triad maps keep their own coordinate data because they do not
represent segmentation cells. Live, unsaved phenotype previews must be saved
before a density map can follow their cells.

Manual assignment preserves **Colour by**, filters, and draw mode. With **Single
colour**, a **Manual labels** checkbox adds assigned-label colours while keeping
unassigned cell outlines visible. Turning it off leaves the ordinary outlines.
Both Colour by controls reflect the same state. A label target is independent
of the display column; **Show** explicitly reveals its manual-label overlay.

In the clicked **Cell** card:

- **Fields** has a search box and checkboxes for the extra phenotype/label rows.
  **All matching fields** affects only the search results. The main phenotype
  stays visible; preferences carry across cells and browser sessions.
- **Markers → Marker sets** uses the named sets from Image layers, with their
  exact names. **Show top** defaults to five, accepts another positive integer,
  and ranks the current displayed values. **All** displays the full selected
  set. These choices are remembered and do not change image channels or the
  calculations behind the values (including Z2's marker population).
- Suggestion provenance and labels needing more training examples are under
  **Details**. Queued assignments and unavailable marker counts stay visible.

Regression coverage includes real browser checks in `viewer_cell_context_browser.py`,
`viewer_cell_card_browser.py`, `viewer_cell_rows_browser.py` and
`viewer_cell_selection_browser.py`, the compact layout in
`viewer_cells_workspace_browser.py`, and CSV import/source isolation in
`viewer_import_choices_browser.py`. Stable per-segmentation import bindings are
checked in `test_viewer_import_bindings.py`, alongside the density API tests in
`test_viewer.py`.

Standing product requirements are maintained in [AGENTS.md](../AGENTS.md) and
the [Studio skill](../skills/spacemap-studio/SKILL.md).
