"""Score the configured local model on synthetic Studio requests (no project data).

Run deliberately on a node where the user's selected model is installed.
Ordinary pytest uses mocked model replies; this is a separate integration run.
"""

import json
import time

CASES = [
    ("Bring up the image called P11_roi_2.", "open_image", {"image": "P11_roi_2"}),
    ("I want to look at P12_roi_1 instead.", "open_image", {"image": "P12_roi_1"}),
    ("Let me see only the CD8 signal.", "show_markers", {"markers": ["CD8"]}),
    ("Could the image show CD3 together with CD8?", "show_markers", {"markers": ["CD3", "CD8"]}),
    ("Use our saved Immune marker combination.", "apply_marker_set", {"set": "Immune"}),
    ("Give each cell its colour from phenotype_ref1.", "colour_by", {"column": "phenotype_ref1"}),
    ("Use phenotype_ml_consensus as the cell colour legend.", "colour_by", {"column": "phenotype_ml_consensus"}),
    ("Where are the CD8 cells most crowded in this image?", "density", {"phenotype": "CD8"}),
    ("Map the local concentration of B cells.", "density", {"phenotype": "B"}),
    ("I would like a density map of CD8 cells on P12_roi_1.", "density", {"phenotype": "CD8", "image": "P12_roi_1"}),
    ("What is the number of CD8 cells here?", "count_cells", {"phenotype": "CD8", "scope": "image"}),
    ("Tell me how many B cells the whole project contains.", "count_cells", {"phenotype": "B", "scope": "project"}),
    ("Give me a breakdown of all cells by phenotype in this image.", "count_cells", {"scope": "image"}),
    ("Take me close to cell number 42.", "zoom_to_cell", {"cell": 42}),
    ("I want cell 100 at the centre of the image.", "zoom_to_cell", {"cell": 100}),
    ("The live CD8 gate should be 125 in raw units.", "set_threshold", {"marker": "CD8", "value": 125}),
    ("Change the raw CD3 positivity cutoff to 80.", "set_threshold", {"marker": "CD3", "value": 80}),
    ("Keep these current ROI thresholds in the project.", "save_thresholds", {}),
    ("Download a picture of what I am viewing.", "save_image", {}),
    (
        "Only display B cells according to phenotype_ref1.",
        "filter_cells",
        {"column": "phenotype_ref1", "phenotypes": ["B"]},
    ),
    (
        "Pick cells whose CD8 value is greater than 100 in the raw scale.",
        "select_cells",
        {
            "rule": {
                "scope": "image",
                "join": "all",
                "conditions": [{"marker": "CD8", "transform": "Raw", "op": ">", "value": 100}],
            }
        },
    ),
    ("Compare the selected group with the remaining cells in a heatmap.", "selection_heatmap", {}),
    (
        "Put the selected cells into phenotype_ref1 as CD8 T.",
        "assign_selection",
        {"column": "phenotype_ref1", "phenotype": "CD8 T"},
    ),
    (
        "Find the cells where CD3 has a Z1 value above 1.5 in every ROI.",
        "select_cells",
        {
            "rule": {
                "scope": "project",
                "join": "all",
                "conditions": [{"marker": "CD3", "transform": "Z1", "op": ">", "value": 1.5}],
            }
        },
    ),
]


def main():
    from spacemap_assistant.model_client import LocalModelClient
    from spacemap_assistant.model_runtime import ensure_running
    from spacemap_gui.studio import assistant
    from spacemap_gui.viewer import registry

    ensure_running(on_wait=lambda value: print(str(value), flush=True))
    client = LocalModelClient(timeout_seconds=120)
    # Explicitly force model interpretation to measure it independently of the
    # instant parser. These substitutions affect only this evaluation process.
    assistant.parse = lambda _: None
    registry.session_view = lambda *args, **kwargs: {"spec": {"layers": []}}
    assistant.viewer_context = lambda _: {
        "image": "P11_roi_2",
        "images": ["P11_roi_2", "P12_roi_1"],
        "markers": ["CD3", "CD8", "CD20"],
        "columns": ["phenotype_ml_consensus", "phenotype_ref1"],
        "phenotypes": ["CD8", "B"],
        "marker_sets": ["Immune"],
    }
    results = []
    for text, name, args in CASES:
        start = time.perf_counter()
        try:
            reply = assistant.propose("synthetic-evaluation", text, client=client)
            command = reply.get("command", {})
            got = command.get("args", {})
            okay = command.get("name") == name and all(
                (set(got.get(key, [])) == set(value) if isinstance(value, list) else got.get(key) == value)
                for key, value in args.items()
            )
            # A model must not add an image or phenotype that was not requested.
            if "image" not in args and "image" in got and got["image"] != "P11_roi_2":
                okay = False
            result = {
                "text": text,
                "expected": {"name": name, "args": args},
                "reply": reply,
                "passed": okay,
                "seconds": round(time.perf_counter() - start, 3),
            }
        except Exception as exc:
            result = {"text": text, "passed": False, "error": str(exc)}
        results.append(result)
        print(json.dumps(result), flush=True)
    score = sum(result["passed"] for result in results) / len(results)
    print(
        json.dumps({"score": score, "passed": sum(result["passed"] for result in results), "total": len(results)}),
        flush=True,
    )
    return 0 if score >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
