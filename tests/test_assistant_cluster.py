from pathlib import Path

import pytest

from spacemap_assistant.cluster import (
    ClusterError,
    ResourceRequest,
    create_probe_script,
    parse_sinfo,
    recommend_partition,
    validate_probe_script,
)


SINFO_SAMPLE = """\
256GBv2*|up|infinite|74|64|256000|(null)|idle
256GBv2*|up|infinite|104|64|256000|(null)|alloc
GPUL4|up|infinite|13|32|512000|gpu:1|idle
GPUL4|up|infinite|3|32|512000|gpu:1|alloc
GPU4A100|up|infinite|5|72|1030000|gpu:4|idle
384GB|up|infinite|17|32|384000|(null)|idle
"""


def test_parse_sinfo_merges_partition_state_rows() -> None:
    partitions = {item.name: item for item in parse_sinfo(SINFO_SAMPLE)}
    assert partitions["256GBv2"].total_nodes == 178
    assert partitions["256GBv2"].idle_nodes == 74
    assert partitions["GPUL4"].gpu_per_node == 1
    assert partitions["GPU4A100"].memory_mib == 1_030_000


def test_recommendation_keeps_cpu_job_off_gpu() -> None:
    selected, reason = recommend_partition(
        ResourceRequest("Leiden", cpus=16, memory_gb=128, gpu_count=0, hours=8),
        parse_sinfo(SINFO_SAMPLE),
    )
    assert selected.name == "256GBv2"
    assert "without reserving a GPU" in reason


def test_recommendation_uses_small_single_gpu_partition() -> None:
    selected, _ = recommend_partition(
        ResourceRequest("Vision", cpus=8, memory_gb=64, gpu_count=1, hours=4),
        parse_sinfo(SINFO_SAMPLE),
    )
    assert selected.name == "GPUL4"


def test_impossible_request_is_rejected() -> None:
    with pytest.raises(ClusterError):
        recommend_partition(
            ResourceRequest("Too large", cpus=256, memory_gb=1024, gpu_count=4, hours=4),
            parse_sinfo(SINFO_SAMPLE),
        )


def test_generated_probe_contains_no_user_command() -> None:
    request = ResourceRequest("Probe; touch /tmp/no", cpus=2, memory_gb=8, gpu_count=0, hours=1)
    path, selected, _ = create_probe_script(request, parse_sinfo(SINFO_SAMPLE))
    script = Path(path).read_text(encoding="utf-8")
    assert selected.name == "256GBv2"
    assert "touch /tmp/no" not in script
    assert "#SBATCH --job-name=spacemap-ai-probe" in script
    assert "SPACEMAP cluster resource probe" in script
    assert "checks passed" in validate_probe_script(path)

