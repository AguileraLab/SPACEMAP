# Updating annotations in an existing cell table

Use this when cell segmentation and measured markers are unchanged, but you have
added or corrected annotation polygons.

1. Save the corrected annotation GeoJSON in the original image's pixel coordinates.
2. In **Cell tagging**, choose **One merged table**, then select the folder and
   existing CSV or Parquet. Tick **Update existing table** beside the table choice.
3. Choose the **Image column** that identifies each cell's image, and select the
   annotation folder or imported GeoJSON source. Check each selected image's file.
4. Press **None** beside the column count to clear the previous column selection.
   Tick only the fields to add or fix: **Inside**, **Piece**, **Size**, or **Distance**.
   Enter new names to add columns. Existing names replace those fields for the
   selected images. Keep the units consistent with existing measurements.
5. Tick the affected images and click **Update: Selected**, or open an image and
   use **This image**. Follow the job in **Jobs**.

The existing file keeps all cells in their original order. Unselected images and
unticked columns keep their values, including marker measurements, corrected
markers, manual references and model predictions. New columns are blank in images
that have not been tagged yet; blank means not processed, not negative.

If a CSV and Parquet with the same base filename both exist, Studio updates them
together after checking their column lists, row counts and cell identities. A
mismatch stops the update; regenerate the CSV copy through Jobs first. Saved
pending labels/results must also be written into the table before retagging.
Backups are retained under `.cell_tagging_backups` in the table's folder, and the
job records which images and columns changed. Without **Update existing table**,
**One merged table** continues to save a separate tagged output.

The parent annotation still defines the tissue pieces used for tagging, even
when its output fields are unticked. If its membership or piece assignments
conflict with existing unticked parent fields, use the original parent polygons
or explicitly include those parent fields in the update.

Batch correction and phenotyping are not rerun automatically. An independent
annotation correction does not require them to run again. Analyses that actually
used a changed field need reviewing—for example, a vessel-distance analysis after
editing vessel polygons, or a tissue-filtered analysis after changing tissue
membership. This workflow does not transfer labels across resegmented cells.
