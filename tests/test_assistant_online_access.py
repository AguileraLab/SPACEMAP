import tempfile
from pathlib import Path

import pytest
from PIL import Image

import spacemap_assistant.online_access as online_access
from spacemap_assistant.paths import (
    AI_ROOT,
    ASSISTANT_DATA_DIR,
    ASSISTANT_PROJECTS_DIR,
    SOURCE_ROOT,
    AssistantPathError,
)


def test_owner_grant_builds_bounded_image_and_table_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSISTANT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as config_dir, tempfile.TemporaryDirectory(
        dir=ASSISTANT_PROJECTS_DIR
    ) as data_dir:
        monkeypatch.setattr(
            online_access,
            "ONLINE_ACCESS_PATH",
            Path(config_dir) / "online_data_access.json",
        )
        folder = Path(data_dir)
        image_path = folder / "public_preview.png"
        csv_path = folder / "public_cells.csv"
        Image.new("RGB", (1800, 900), (120, 30, 20)).save(image_path)
        csv_path.write_text("cell,label\n1,T cell\n2,B cell\n", encoding="utf-8")

        grant = online_access.grant_online_access(folder)
        online_access.save_online_share_limits(2, 25, 512)
        prompt = f"Review the image and table in {folder}"
        planned = online_access.plan_online_evidence(prompt)
        assert len(planned) == 2

        content, manifest = online_access.build_online_evidence_content(
            prompt,
            model_supports_images=True,
        )
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[-1]["type"] == "text"
        assert "T cell" in content[-1]["text"]
        assert str(folder) not in content[-1]["text"]
        assert {item["kind"] for item in manifest} == {"resized_png", "table_sample"}
        image_manifest = next(item for item in manifest if item["kind"] == "resized_png")
        assert max(image_manifest["width"], image_manifest["height"]) <= 512

        assert online_access.revoke_online_access(grant["id"])["path"] == str(folder)
        # Built-in Projects access remains recursive even when a redundant
        # extra grant is removed.
        assert len(online_access.plan_online_evidence(prompt)) == 2


def test_online_access_never_allows_application_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as config_dir:
        monkeypatch.setattr(
            online_access,
            "ONLINE_ACCESS_PATH",
            Path(config_dir) / "online_data_access.json",
        )
        with pytest.raises(AssistantPathError, match="Only project data"):
            online_access.grant_online_access(SOURCE_ROOT)


def test_built_in_projects_and_every_subfolder_are_online_eligible() -> None:
    nested = ASSISTANT_PROJECTS_DIR / "study" / "outputs" / "annotations" / "result.geojson"
    assert online_access.is_online_share_allowed(ASSISTANT_PROJECTS_DIR)
    assert online_access.is_online_share_allowed(nested)
    assert not online_access.is_online_share_allowed(SOURCE_ROOT / "spacemap" / "utils.py")


def test_online_preview_can_describe_cluster_figures_without_polygon_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ASSISTANT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_PROJECTS_DIR) as data_dir:
        heatmap = Path(data_dir) / "step3_cluster_balanced.png"
        cluster_map = Path(data_dir) / "step3_cluster_map.png"
        Image.new("RGB", (100, 60), "white").save(heatmap)
        Image.new("RGB", (100, 60), "black").save(cluster_map)
        note = "Image 1 is a marker heatmap. Image 2 is the spatial cluster map."
        content, manifest = online_access.build_online_preview_content(
            [heatmap, cluster_map],
            "Label the clusters.",
            model_supports_images=True,
            context_note=note,
        )
        assert len(manifest) == 2
        assert note in content[-1]["text"]
        assert "polygon tiles" not in content[-1]["text"]
