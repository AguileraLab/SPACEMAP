from pathlib import Path

from PIL import Image
import numpy as np

import spacemap_assistant.cluster_labeling as cluster_labeling
from spacemap_gui.tabs.tab4_annotation.s3_pixel_classification import _cluster_spatial_summary


def _allow_test_path(path: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(path)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def test_current_pixel_classification_evidence_supplies_tables_and_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(cluster_labeling, "require_assistant_read_path", _allow_test_path)
    assignment = tmp_path / "step3_cluster_assignments.csv"
    profile = tmp_path / "step3_cluster_shrunk_z_values.csv"
    heatmap = tmp_path / "step3_cluster_balanced.png"
    cluster_map = tmp_path / "step3_cluster_map.png"
    spatial = tmp_path / "step3_cluster_spatial_summary.csv"
    assignment.write_text(
        "Cluster,Pixels,% tissue,Reliability,Top markers (shrunk Z),Domain\n"
        '0,1200,12.5,0.91,"PanCK (+2.1), E-cadherin (+1.8)",Unassigned\n'
        '1,700,7.2,0.72,"CD45 (+2.0), CD3 (+1.7)",Unassigned\n',
        encoding="utf-8",
    )
    profile.write_text(
        ",PanCK,E-cadherin,CD45,CD3\n"
        "0,2.1,1.8,-1.0,-0.8\n"
        "1,-0.9,-0.7,2.0,1.7\n",
        encoding="utf-8",
    )
    Image.new("RGB", (24, 16), "white").save(heatmap)
    Image.new("RGB", (24, 16), "black").save(cluster_map)
    spatial.write_text(
        "Cluster,Map location,Centroid X (%),Centroid Y (%),Horizontal span (%),Vertical span (%),Touches map edge,Adjacent clusters\n"
        '0,top-left,20.0,10.0,30.0,20.0,True,"1"\n'
        '1,bottom-right,80.0,90.0,20.0,10.0,True,"0"\n',
        encoding="utf-8",
    )
    state = {
        cluster_labeling.PAGE_EVIDENCE_KEY: {
            "tab4": {
                "section": "pixel_classification",
                "sample_id": "sample_A",
                "assignment_csv": str(assignment),
                "profile_tables": {"shrunk_z": str(profile), "spatial": str(spatial)},
                "figures": {
                    "balanced_heatmap": str(heatmap),
                    "cluster_map": str(cluster_map),
                },
            }
        }
    }

    result = cluster_labeling.prepare_cluster_labeling_evidence(
        "Can you label these clusters from their marker expression?",
        state,
    )

    assert result is not None
    assert result["profile_csv"] == profile
    assert result["images"] == [heatmap, cluster_map]
    assert len(result["clusters"]) == 2
    assert "PanCK +2.10" in result["clusters"][0]["enriched"]
    assert "location=top-left" in result["clusters"][0]["spatial"]
    assert "exact data behind the heatmap" in result["note"]
    assert "do not ask for a CSV" in result["note"]


def test_explicit_assignment_csv_works_without_live_tab_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(cluster_labeling, "require_assistant_read_path", _allow_test_path)
    assignment = tmp_path / "step3_cluster_assignments.csv"
    assignment.write_text(
        "Cluster,Pixels,% tissue,Reliability,Top markers (shrunk Z),Domain\n"
        '4,50,0.5,0.30,"Vimentin (+1.4)",Stroma\n',
        encoding="utf-8",
    )

    result = cluster_labeling.prepare_cluster_labeling_evidence(
        f"Help label the clusters using marker expression {assignment}",
        {},
    )

    assert result is not None
    assert result["assignment_csv"] == assignment
    assert result["clusters"][0]["cluster"] == "4"
    assert result["clusters"][0]["current_domain"] == "Stroma"


def test_unrelated_question_does_not_activate_cluster_adapter() -> None:
    assert cluster_labeling.prepare_cluster_labeling_evidence(
        "What is the difference between these two directories?",
        {},
    ) is None


def test_spatial_summary_describes_location_span_edges_and_neighbors() -> None:
    cluster_map = np.array(
        [
            [0, 0, -1, -1],
            [0, 0, 1, 1],
            [-1, 1, 1, 1],
            [-1, 1, 1, 1],
        ]
    )
    summary = _cluster_spatial_summary(cluster_map, [0, 1]).set_index("Cluster")
    assert summary.loc[0, "Map location"] == "top-left"
    assert bool(summary.loc[0, "Touches map edge"])
    assert summary.loc[0, "Adjacent clusters"] == "1"
    assert summary.loc[1, "Adjacent clusters"] == "0"
