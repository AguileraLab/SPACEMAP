"""The checked requests available to Studio buttons and the assistant."""

from dataclasses import dataclass, field
import json
import math


@dataclass(frozen=True)
class Argument:
    kind: str
    required: bool = False
    default: object = None
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    allow_empty: bool = False

    def read(self, name, value):
        good = {
            "text": lambda v: isinstance(v, str) and 0 < len(v.strip()) <= 512,
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "names": lambda v: (
                isinstance(v, list)
                and (0 if self.allow_empty else 1) <= len(v) <= 128
                and all(isinstance(s, str) and 0 < len(s.strip()) <= 512 for s in v)
            ),
            "rule": lambda v: isinstance(v, dict),
            "settings": lambda v: (
                isinstance(v, dict)
                and 0 < len(v) <= 60
                and all(isinstance(k, str) and 0 < len(k) <= 64 for k in v)
                and len(json.dumps(v, default=str)) <= 20000
            ),
            "labels": lambda v: (
                isinstance(v, dict)
                and 0 < len(v) <= 256
                and all(
                    isinstance(k, str)
                    and isinstance(n, str)
                    and 0 < len(k.strip()) <= 512
                    and 0 < len(n.strip()) <= 512
                    for k, n in v.items()
                )
            ),
            "points": lambda v: (
                isinstance(v, list)
                and 3 <= len(v) <= 2048
                and all(
                    isinstance(p, list)
                    and len(p) == 2
                    and all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in p)
                    for p in v
                )
            ),
            "levels": lambda v: (
                isinstance(v, dict)
                and 0 < len(v) <= 256
                and all(
                    isinstance(k, str)
                    and 0 < len(k.strip()) <= 512
                    and isinstance(n, (int, float))
                    and not isinstance(n, bool)
                    and math.isfinite(n)
                    and 0 <= n <= 1e9
                    for k, n in v.items()
                )
            ),
            "ids": lambda v: (
                isinstance(v, list)
                and 0 < len(v) <= 100000
                and all(isinstance(i, int) and not isinstance(i, bool) and 0 < i < 2**32 for i in v)
            ),
        }[self.kind](value)
        if not good:
            raise ValueError(f"Check {name}: expected {self.kind}.")
        if self.choices and value not in self.choices:
            raise ValueError(f"Choose {', '.join(map(str, self.choices))} for {name}.")
        if self.minimum is not None and value < self.minimum or self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} is outside its allowed range.")
        return value.strip() if isinstance(value, str) else value


@dataclass(frozen=True)
class Command:
    description: str
    arguments: dict[str, Argument] = field(default_factory=dict)
    changes_project: bool = False
    undo: str = "Restores the previous view."

    def validate(self, values):
        if not isinstance(values, dict):
            raise ValueError("Command arguments must be named values.")
        extra = set(values) - set(self.arguments)
        if extra:
            raise ValueError("Unknown arguments: " + ", ".join(sorted(extra)))
        result = {}
        for name, argument in self.arguments.items():
            if name in values:
                result[name] = argument.read(name, values[name])
            elif argument.required:
                raise ValueError(f"Provide {name}.")
            elif argument.default is not None:
                result[name] = argument.default
        return result


TEXT = Argument("text")
REQUIRED_TEXT = Argument("text", required=True)
IMAGE = {"image": TEXT}
COLUMN = {"column": TEXT}
COMMANDS = {
    "open_image": Command("Open a project image.", {"image": REQUIRED_TEXT}),
    "show_markers": Command(
        "Show the selected image markers.",
        {**IMAGE, "markers": Argument("names", required=True), "replace": Argument("boolean", default=True)},
    ),
    "apply_marker_set": Command("Apply a saved marker set.", {**IMAGE, "set": REQUIRED_TEXT}),
    "colour_by": Command("Colour cells using a table column.", {**IMAGE, "column": REQUIRED_TEXT}),
    "filter_cells": Command(
        "Show selected phenotype groups.",
        {**IMAGE, "column": REQUIRED_TEXT, "phenotypes": Argument("names", required=True)},
    ),
    "density": Command(
        "Show the density of a phenotype group; with table (a density table such as Tab 8's triads), of one of its groups.",
        {
            **IMAGE,
            "column": TEXT,
            "phenotype": REQUIRED_TEXT,
            "table": TEXT,
            "radius": Argument("number", default=40, minimum=1, maximum=5000),
        },
    ),
    "zoom_to_cell": Command(
        "Centre the view on a cell.",
        {**IMAGE, "cell": Argument("integer", required=True, minimum=1, maximum=2**32 - 1)},
    ),
    "count_cells": Command(
        "Count cells by phenotype.",
        {
            **IMAGE,
            **COLUMN,
            "phenotype": TEXT,
            "scope": Argument("text", default="image", choices=("image", "project")),
        },
        undo="No data changed.",
    ),
    "set_threshold": Command(
        "Set a live RESOLVE threshold in raw units.",
        {**IMAGE, "marker": REQUIRED_TEXT, "value": Argument("number", required=True, minimum=0, maximum=65535)},
        undo="Restores the previous live threshold.",
    ),
    "save_thresholds": Command(
        "Save the current ROI thresholds.",
        IMAGE,
        changes_project=True,
        undo="Restores the saved threshold file if it has not changed again.",
    ),
    "assign_phenotype": Command(
        "Assign selected cell IDs to a reference column.",
        {**IMAGE, "cells": Argument("ids", required=True), "column": REQUIRED_TEXT, "phenotype": REQUIRED_TEXT},
        changes_project=True,
        undo="Restores previous labels for these cells if they have not changed again.",
    ),
    "save_image": Command(
        "Save the current view as an image.", {}, undo="The downloaded file can be removed by its owner."
    ),
    "change_settings": Command(
        "Change a step's saved settings, the ones its Run uses (shared with the classic tab). method: leiden, som, "
        "vortex, consensus, subphenotyping, neighbourhoods, interactions or batch_correction; changes: {setting: new value} using "
        "the names and kinds of values listed in SETTINGS.",
        {
            "method": Argument(
                "text",
                required=True,
                choices=(
                    "leiden",
                    "som",
                    "vortex",
                    "consensus",
                    "subphenotyping",
                    "neighbourhoods",
                    "interactions",
                    "batch_correction",
                ),
            ),
            "changes": Argument("settings", required=True),
        },
        changes_project=True,
        undo="Restores the previous values if they have not changed again.",
    ),
    "run_batch_correction": Command(
        "Run Tab 6's batch correction with its saved settings and markers on every ROI's cell table, as a job.",
        {},
        changes_project=True,
        undo="The corrected table is a new file; the ROI tables and earlier corrected tables are kept.",
    ),
    "run_resolve_all": Command(
        "Run RESOLVE for every ROI using the saved rules and positivity method.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False), "column": REQUIRED_TEXT},
        changes_project=True,
        undo="Stop the job before it saves, or keep the separate result column.",
    ),
    "run_mr_ml": Command(
        "Train MR-ML from a manual reference column and save new predictions.",
        {**IMAGE, "overwrite": Argument("boolean", default=False), "column": TEXT, "reference_column": TEXT,
         "settings_revision": TEXT,
         "methods": Argument("names", default=[], allow_empty=True),
         "confidence_column": TEXT, "refined_column": TEXT,
         "confidence_threshold": Argument("number", default=0.75, minimum=0, maximum=1),
         "low_confidence_label": Argument("text", default="Others"),
         "excluded_phenotypes": Argument("names", default=["Others"], allow_empty=True),
         "morphology_cols": Argument("names", default=[], allow_empty=True)},
        changes_project=True,
        undo="Earlier predictions and scores are kept unless Overwrite existing results is selected.",
    ),
    "run_leiden": Command(
        "Run Tab 7's Leiden clustering with its saved settings on the combined table, as a job.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False)},
        changes_project=True,
        undo="Earlier results are kept unless overwrite is explicitly selected; overwrite replaces this output and its cluster column.",
    ),
    "train_cellpose": Command(
        "Train the next version of the session's Cellpose model on the corrected training tiles, as a job.",
        {**IMAGE, "epochs": Argument("integer", default=100, minimum=1, maximum=2000)},
        changes_project=True,
        undo="Each version is saved under its own name; earlier versions are kept.",
    ),
    "run_som": Command(
        "Run Tab 7's SOM clustering with its saved settings on the combined table, as a job.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False)},
        changes_project=True,
        undo="Earlier results are kept unless Overwrite existing results is selected.",
    ),
    "run_vortex": Command(
        "Run Tab 7's VORTEX clustering with its saved settings on the combined table, as a job.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False)},
        changes_project=True,
        undo="Earlier results are kept unless Overwrite existing results is selected.",
    ),
    "run_subphenotyping": Command(
        "Split the chosen phenotypes into sub-clusters with Tab 7's sub-phenotyping (mode discover finds the markers first), as a job.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False), "mode": Argument("text", default="run", choices=("discover", "run"))},
        changes_project=True,
        undo="Earlier sub-clusters are kept unless Overwrite existing results is selected.",
    ),
    "run_consensus_ml": Command(
        "Run Tab 7's Consensus ML with its saved settings: methods that agree train a model that labels every cell, as a job.",
        {**IMAGE, "settings_revision": TEXT, "overwrite": Argument("boolean", default=False)},
        changes_project=True,
        undo="Earlier results and scores are kept unless Overwrite existing results is selected.",
    ),
    "run_neighbourhoods": Command(
        "Find Tab 8's cellular neighbourhoods with its saved settings on the combined table (or only the images named), as a job.",
        {**IMAGE, "images": Argument("names"), "settings_revision": TEXT, "overwrite": Argument("boolean", default=False)},
        changes_project=True,
        undo="Earlier neighbourhood groups and names are kept unless Overwrite existing results is selected.",
    ),
    "compare_pair": Command(
        "Open a registration pair: its CODEX image, with its H&E over it to align by eye (turn, scale, move).",
        {"sample": REQUIRED_TEXT},
    ),
    "show_segmentation_region": Command(
        "Show the open ROI's annotation file (the one its segmentation region and cell tags use) on the image.",
        {**IMAGE},
    ),
    "tag_cells": Command(
        "Tag each cell with the annotations it sits in (membership, ROI, size, distance, bed, edge), with Tab 5's "
        "saved cell tagging settings, as a job. Writes tagged copies of the cell tables.",
        {"tables": Argument("names")},
        changes_project=True,
        undo="The original cell tables are unchanged; tagged tables are new files (a table tagged again is replaced).",
    ),
    "run_cell_interactions": Command(
        "Test which phenotypes are neighbours more or less often than by chance, with Tab 8's saved settings (all images, or only those named), as a job.",
        {**IMAGE, "images": Argument("names")},
        changes_project=True,
        undo="Nothing in the table changes; the result stays with the job.",
    ),
    "name_clusters": Command(
        "Apply or change cluster names for a finished clustering job. Preserve other names; clear removes chosen names, replace replaces the complete mapping. Update this run’s phenotype column.",
        {**IMAGE, "job_id": REQUIRED_TEXT, "names": Argument("labels"), "clear":Argument("names"), "replace":Argument("boolean")},
        changes_project=True,
        undo="Earlier name mappings remain in the saved jobs; edits update this run’s output column.",
    ),
    "train_reference_model": Command(
        "Train and compare two saved suggestion models from a manual reference or another phenotype column.",
        {**IMAGE, "reference_column": REQUIRED_TEXT, "checked_only": Argument("boolean", default=False)},
        changes_project=True,
        undo="The models are saved separately and no cell labels are changed.",
    ),
    "save_background_levels": Command(
        "Save this image's background level per marker as Tab 2's threshold table.",
        {**IMAGE, "levels": Argument("levels", required=True), "method": TEXT},
        changes_project=True,
        undo="Restores the previous threshold table if it has not changed again.",
    ),
    "subtract_background": Command(
        "Write background-removed images from saved levels, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all"))},
        changes_project=True,
        undo="The originals stay; the background-removed images are separate files.",
    ),
    "open_background_image": Command(
        "Open an image of the Background step with its annotation file, to measure inside the chosen class.",
        {"image": REQUIRED_TEXT},
    ),
    "detect_background_levels": Command(
        "Detect each listed image's background level per marker (Tab 2's method, whole image or inside its annotation class), kept in the project, as a job.",
        {"images": Argument("names", required=True), "method": Argument("text", default="Triangle"), "inside": Argument("boolean", default=True),
         "percentile": Argument("number", default=95, minimum=0, maximum=100),
         "manual": Argument("number", default=0), "native": Argument("boolean", default=False)},
        changes_project=True,
        undo="The levels kept in the project are replaced; exported threshold tables are not changed.",
    ),
    "marker_regions": Command(
        "Find the regions where the chosen markers are above their levels (the saved marker-region settings) and save them as the image's regions to review; the open image at once, the ticked ones as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image",)), "images": Argument("names")},
        changes_project=True,
        undo="Replaces each image's regions file (names given to its regions are lost); detect again to redo.",
    ),
    "open_annotation_image": Command(
        "Open an image of the Annotation step with its tissue and the artifacts drawn by hand, to check and draw more.",
        {"image": REQUIRED_TEXT},
    ),
    "open_roi_image": Command(
        "Open an image of the ROI step with its ROIs as a layer, to draw, name and check them.",
        {"image": REQUIRED_TEXT},
    ),
    "detect_roi_cores": Command(
        "Find the TMA cores of the listed images (classic Tab 3's detector and the saved core settings) and save them as their ROIs; the open image at once, the others as a job.",
        {"images": Argument("names", required=True), "open": TEXT},
        changes_project=True,
        undo="Each image's ROI file is replaced by its cores; delete or redraw them on the image.",
    ),
    "extract_roi_images": Command(
        "Cut out the ROIs of the listed images as OME-TIFF images (all of the parent image's metadata kept), as a job.",
        {"images": Argument("names", required=True), "keep_outside": Argument("boolean", default=False)},
        changes_project=True,
        undo="ROIs already extracted are kept; new ROI images are separate files.",
    ),
    "export_background": Command(
        "Export the listed images' background levels as thresholds (CSV) and/or background-removed images (OME-TIFF), as a job.",
        {"images": Argument("names", required=True), "csv": Argument("boolean", default=True), "write_image": Argument("boolean", default=False),
         "use_blanks": Argument("boolean", default=True), "use_levels": Argument("boolean", default=True), "open": TEXT,
         "levels": Argument("levels"), "method": TEXT},
        changes_project=True,
        undo="The originals stay; threshold tables and images are separate files (a table exported again is replaced).",
    ),
    "draw_rois": Command(
        "Open this image's Tab 3 ROI file as a layer to draw and name ROIs.",
        IMAGE,
        undo="Removes the ROI layer from the view; drawn ROIs stay saved.",
    ),
    "extract_rois": Command(
        "Cut out ROI images with Tab 3, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "keep_outside": Argument("boolean", default=False)},
        changes_project=True,
        undo="ROIs already extracted are kept; new ROI images are separate files.",
    ),
    "open_segmentation_image": Command("Open an image of the Segmentation step with its cells.", {"image": REQUIRED_TEXT}),
    "segment_images": Command(
        "Segment cells with Tab 5's saved settings, as a job; shape and ids: only inside polygons picked on the open image.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "images", "all")), "shape": TEXT,
         "ids": Argument("names", default=[], allow_empty=True), "images": Argument("names", default=[], allow_empty=True),
         "again": Argument("boolean", default=False), "tag": Argument("boolean")},
        changes_project=True,
        undo="Images already segmented are kept; new results are separate files.",
    ),
    "register_pairs": Command(
        "Register CODEX and H&E pairs with Tab 1, as a job.",
        {
            **IMAGE,
            "scope": Argument("text", default="image", choices=("image", "all", "pairs")),
            "samples": Argument("names"),
            "again": Argument("boolean", default=False),
            "anchor": TEXT,
            "rotation": Argument("integer", default=0, choices=(0, 90, 180, 270)),
            "keypoints": Argument("integer", default=10000, minimum=500, maximum=100000),
        },
        changes_project=True,
        undo="Pairs already registered are kept; registering one again first moves its merged image into previous/.",
    ),
    "detect_tissue": Command(
        "Detect tissue with Tab 4's step 1 and save its annotation files, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names")},
        changes_project=True,
        undo="Replaces Tab 4's tissue files for these images; run it again with other settings to redo.",
    ),
    "clean_tissue": Command(
        "Remove blood clots, blur, bright spots and lint from the valid tissue with Tab 4, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names")},
        changes_project=True,
        undo="Replaces Tab 4's tissue and artifact files for these images; run it again with other settings to redo.",
    ),
    "detect_artifacts": Command(
        "Detect artifacts as editable polygons, keeping the tissue unchanged.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names")},
        changes_project=True,
        undo="Earlier artifacts are kept in the layer's History.",
    ),
    "subtract_artifacts": Command(
        "Subtract reviewed artifacts from the selected Tissue annotation and save it.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names")},
        changes_project=True,
        undo="Earlier Tissue versions remain available in History.",
    ),
    "apply_pieces": Command(
        "Split the valid tissue into pieces with Tab 4's step 2b and keep the pieces that pass, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names"), "tissue_version": Argument("text", default="original", choices=("original", "cleaned"))},
        changes_project=True,
        undo="Replaces Tab 4's valid tissue and piece files for these images; a later run starts again from all pieces.",
    ),
    "make_zones": Command(
        "Pool domains and build zones with Tab 4's step 4 and the saved rules, as a job.",
        {**IMAGE, "scope": Argument("text", default="image", choices=("image", "all")), "images": Argument("names")},
        changes_project=True,
        undo="Replaces Tab 4's zones file for these images; run it again with other rules to redo.",
    ),
    "export_classes": Command(
        "Write Tab 4's class and domain files for the open image with the classes named in Studio, as a job.",
        {**IMAGE},
        changes_project=True,
        undo="Replaces Tab 4's class and domain files for this image; export again to redo.",
    ),
    "show_job_result": Command("Show a completed job as a cell layer.", {"job_id": REQUIRED_TEXT}),
    "select_cells": Command("Select cells using marker rules.", {**IMAGE, "rule": Argument("rule", required=True)}),
    "refine_selection": Command(
        "Add or remove cells from the current selection.",
        {
            "cells": Argument("ids"),
            "points": Argument("points"),
            "mode": Argument("text", default="toggle", choices=("add", "remove", "toggle")),
        },
    ),
    "selection_heatmap": Command("Compare selected cells with other cells.", {}, undo="No data changed."),
    "assign_selection": Command(
        "Assign the current selection to a reference column.",
        {"column": TEXT, "phenotype": REQUIRED_TEXT},
        changes_project=True,
        undo="Restores the previous labels if they have not changed again.",
    ),
    "save_rule": Command(
        "Save the current rule in this project.",
        {"rule_name": REQUIRED_TEXT},
        changes_project=True,
        undo="The saved rule stays in the project library.",
    ),
    "apply_rule": Command(
        "Apply a saved rule to this image or all ROIs.",
        {**IMAGE, "rule_name": REQUIRED_TEXT, "scope": Argument("text", default="image", choices=("image", "project"))},
    ),
    "undo": Command("Undo a command.", {"command_id": REQUIRED_TEXT}, undo="An undo cannot be undone again."),
}


def catalogue():
    return {
        name: {
            "description": command.description,
            "changes_project": command.changes_project,
            "undo": command.undo,
            "arguments": {
                key: {
                    "type": arg.kind,
                    "required": arg.required,
                    "default": arg.default,
                    **({"choices": list(arg.choices)} if arg.choices else {}),
                }
                for key, arg in command.arguments.items()
            },
        }
        for name, command in COMMANDS.items()
    }


def validate(name, args):
    if name not in COMMANDS:
        raise ValueError("Unknown command. Choose one of the listed Studio actions.")
    return COMMANDS[name].validate(args)
