import pytest

from spacemap_assistant.general_action import (
    GeneralActionError,
    parse_job_envelope,
    validate_generated_code,
)
from spacemap_assistant.paths import ASSISTANT_PROJECTS_DIR


PROJECT = ASSISTANT_PROJECTS_DIR / "AI_test"


def test_valid_scientific_job_uses_injected_paths() -> None:
    validate_generated_code(
        """
from pathlib import Path
from spacemap.utils import extract_rois_with_geometry_from_geojson
import json
import pandas as pd

payload = json.loads(Path(INPUTS["geojson"]).read_text())
rows = extract_rois_with_geometry_from_geojson(payload)
pd.DataFrame({"roi_count": [len(rows)]}).to_csv(OUTPUT_DIR / "summary.csv", index=False)
""",
        PROJECT,
    )


@pytest.mark.parametrize(
    "code, message",
    [
        ("import os\nfrom spacemap import utils\nprint(OUTPUT_DIR)", "Import 'os'"),
        (
            "from spacemap import utils\nopen(OUTPUT_DIR / 'x.txt', 'w')",
            "Call 'open'",
        ),
        (
            "from spacemap import utils\nwhile True:\n    print(OUTPUT_DIR)",
            "while loops",
        ),
        (
            "from spacemap import utils\nOUTPUT_DIR = '/tmp'",
            "protected input/output variables",
        ),
    ],
)
def test_generated_job_rejects_unsafe_language(code: str, message: str) -> None:
    with pytest.raises(GeneralActionError, match=message):
        validate_generated_code(code, PROJECT)


def test_generated_job_rejects_absolute_path_outside_project() -> None:
    code = "from spacemap import utils\nprint('/etc/passwd', OUTPUT_DIR)"
    with pytest.raises(GeneralActionError, match="outside the selected project"):
        validate_generated_code(code, PROJECT)


def test_job_envelope_is_removed_from_user_visible_answer() -> None:
    answer = """I prepared a job.
<spacemap_job>
{"title":"Example","inputs":{"csv":"/x"},"parameters":{},"code":"x","output_dir":"/y","overwrite":false}
</spacemap_job>"""
    clean, payload = parse_job_envelope(answer)
    assert clean == "I prepared a job."
    assert payload is not None
    assert payload["title"] == "Example"


def test_incomplete_job_envelope_is_never_treated_as_chat_text() -> None:
    with pytest.raises(GeneralActionError, match="incomplete"):
        parse_job_envelope('Plan follows.\n<spacemap_job>\n{"title": "cut off"')
