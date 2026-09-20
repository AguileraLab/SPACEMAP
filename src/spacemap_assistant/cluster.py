"""Small, allowlisted bridge between the SPACEMAP assistant and a Slurm cluster.

This is optional.  It is useful when SPACEMAP runs on the login node of an HPC
cluster, and is inert everywhere else: every entry point raises a readable
error when ``sinfo`` and ``sbatch`` are not installed.

The module deliberately does not accept terminal commands from chat.  It can
inspect the cluster, recommend a partition, create one fixed resource-probe
job, and submit only a probe script that it generated and validated itself.

Nothing here is specific to one site.  Partitions are ranked by what Slurm
reports -- idle nodes, memory, whether they carry GPUs -- and a site that wants
particular partitions tried first can name them in the assistant's cluster
settings.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spacemap.ai_settings import load_settings

from .paths import ASSISTANT_DATA_DIR, require_assistant_write_path


SINFO_FORMAT = "%P|%a|%l|%D|%c|%m|%G|%t"
SAFE_JOB_NAME = "spacemap-ai-probe"
def preferred_partitions() -> list[str]:
    """Partition names this site wants tried first, most preferred first.

    Configured in Tab 0 under the assistant's cluster settings.  When it is
    empty -- the usual case -- partitions are ranked purely by what Slurm
    reports, which works on any cluster.
    """
    raw = str(load_settings().get("cluster", {}).get("partition") or "")
    return [name.strip() for name in raw.replace(",", " ").split() if name.strip()]


class ClusterError(RuntimeError):
    """A safe, user-readable cluster failure."""


@dataclass(frozen=True)
class PartitionInfo:
    name: str
    available: bool
    time_limit: str
    total_nodes: int
    idle_nodes: int
    allocated_nodes: int
    cpus_per_node: int
    memory_mib: int
    gpu_per_node: int
    states: tuple[str, ...]

    @property
    def has_gpu(self) -> bool:
        return self.gpu_per_node > 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["states"] = ", ".join(self.states)
        payload["memory_gib"] = round(self.memory_mib / 1024, 1)
        return payload


@dataclass(frozen=True)
class ResourceRequest:
    workload: str
    cpus: int
    memory_gb: int
    gpu_count: int
    hours: int
    partition: str = "auto"

    def validated(self) -> "ResourceRequest":
        workload = str(self.workload).strip() or "Resource check"
        cpus = int(self.cpus)
        memory_gb = int(self.memory_gb)
        gpu_count = int(self.gpu_count)
        hours = int(self.hours)
        partition = str(self.partition).strip() or "auto"
        if not 1 <= cpus <= 256:
            raise ValueError("CPU count must be between 1 and 256.")
        if not 1 <= memory_gb <= 1024:
            raise ValueError("Memory must be between 1 and 1024 GB.")
        if not 0 <= gpu_count <= 4:
            raise ValueError("GPU count must be between 0 and 4.")
        if not 1 <= hours <= 168:
            raise ValueError("Time must be between 1 hour and 7 days.")
        if partition != "auto" and not re.fullmatch(r"[A-Za-z0-9_-]+", partition):
            raise ValueError("Partition name contains unsupported characters.")
        return ResourceRequest(workload, cpus, memory_gb, gpu_count, hours, partition)


WORKLOAD_PROFILES: dict[str, ResourceRequest] = {
    "GPU segmentation": ResourceRequest("GPU segmentation", 16, 128, 1, 8),
    "Leiden parameter search": ResourceRequest("Leiden parameter search", 16, 128, 0, 8),
    "Large image / high-memory work": ResourceRequest("Large image / high-memory work", 32, 384, 0, 12),
    "Cluster resource probe": ResourceRequest("Cluster resource probe", 2, 8, 0, 1),
}


def _run(args: list[str], *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(args[0])
    if executable is None:
        raise ClusterError(
            f"'{args[0]}' was not found. Cluster features need Slurm client "
            "commands on PATH; they are unavailable on this machine."
        )
    try:
        result = subprocess.run(
            [executable, *args[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ClusterError(f"The cluster command could not finish: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise ClusterError(detail[:1000])
    return result


def _first_integer(value: str) -> int:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def _gpu_count(gres: str) -> int:
    if not gres or gres == "(null)":
        return 0
    counts = [int(value) for value in re.findall(r"gpu(?::[^:,()]+)?:(\d+)", gres)]
    return max(counts, default=0)


def parse_sinfo(text: str) -> list[PartitionInfo]:
    """Parse the stable, pipe-delimited ``sinfo`` format and merge state rows."""
    grouped: dict[str, dict[str, Any]] = {}
    for raw_line in str(text).splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) != 8:
            continue
        raw_name, availability, limit, nodes, cpus, memory, gres, state = fields
        name = raw_name.rstrip("*")
        if not name:
            continue
        node_count = _first_integer(nodes)
        normalized_state = state.rstrip("*").lower()
        record = grouped.setdefault(
            name,
            {
                "available": availability.lower() == "up",
                "time_limit": limit,
                "total_nodes": 0,
                "idle_nodes": 0,
                "allocated_nodes": 0,
                "cpus_per_node": 0,
                "memory_mib": 0,
                "gpu_per_node": 0,
                "states": set(),
            },
        )
        record["available"] = bool(record["available"] or availability.lower() == "up")
        record["total_nodes"] += node_count
        if normalized_state == "idle":
            record["idle_nodes"] += node_count
        if normalized_state in {"alloc", "allocated", "mix", "mixed"}:
            record["allocated_nodes"] += node_count
        record["cpus_per_node"] = max(record["cpus_per_node"], _first_integer(cpus))
        record["memory_mib"] = max(record["memory_mib"], _first_integer(memory))
        record["gpu_per_node"] = max(record["gpu_per_node"], _gpu_count(gres))
        record["states"].add(normalized_state)

    return [
        PartitionInfo(
            name=name,
            available=record["available"],
            time_limit=record["time_limit"],
            total_nodes=record["total_nodes"],
            idle_nodes=record["idle_nodes"],
            allocated_nodes=record["allocated_nodes"],
            cpus_per_node=record["cpus_per_node"],
            memory_mib=record["memory_mib"],
            gpu_per_node=record["gpu_per_node"],
            states=tuple(sorted(record["states"])),
        )
        for name, record in sorted(grouped.items())
    ]


def query_partitions() -> tuple[list[PartitionInfo], str]:
    args = ["sinfo", "--noheader", f"--format={SINFO_FORMAT}"]
    result = _run(args)
    return parse_sinfo(result.stdout), "$ " + " ".join(args) + "\n" + result.stdout.strip()


def query_my_jobs() -> tuple[list[dict[str, str]], str]:
    args = ["squeue", "--me", "--noheader", "--format=%i|%P|%j|%T|%M|%D|%R"]
    result = _run(args)
    jobs: list[dict[str, str]] = []
    keys = ["job_id", "partition", "name", "state", "elapsed", "nodes", "location"]
    for line in result.stdout.splitlines():
        values = line.split("|", 6)
        if len(values) == len(keys):
            jobs.append(dict(zip(keys, values)))
    terminal = "$ " + " ".join(args) + "\n" + (result.stdout.strip() or "No active jobs.")
    return jobs, terminal


def recommend_partition(request: ResourceRequest, partitions: list[PartitionInfo]) -> tuple[PartitionInfo, str]:
    request = request.validated()
    needed_mib = request.memory_gb * 1024
    candidates = [
        partition
        for partition in partitions
        if partition.available
        and partition.cpus_per_node >= request.cpus
        and partition.memory_mib >= needed_mib
        and (partition.gpu_per_node >= request.gpu_count if request.gpu_count else not partition.has_gpu)
    ]
    if request.partition != "auto":
        candidates = [partition for partition in candidates if partition.name == request.partition]
    if not candidates:
        kind = "GPU" if request.gpu_count else "CPU/high-memory"
        raise ClusterError(
            f"No active {kind} partition matches this request. Try a smaller "
            "request or choose a partition manually."
        )

    preference = preferred_partitions()
    preference_rank = {name: index for index, name in enumerate(preference)}
    candidates.sort(
        key=lambda item: (
            0 if item.idle_nodes else 1,
            preference_rank.get(item.name, len(preference)),
            item.memory_mib,
            -item.idle_nodes,
        )
    )
    selected = candidates[0]
    reason = (
        f"{selected.name} matches {request.cpus} CPU(s), {request.memory_gb} GB RAM"
        + (f", and {request.gpu_count} GPU(s)" if request.gpu_count else " without reserving a GPU")
        + f". Slurm reports {selected.idle_nodes} idle node(s) in this partition right now."
    )
    return selected, reason


def _safe_probe_script(request: ResourceRequest, partition: PartitionInfo) -> str:
    request = request.validated()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", partition.name):
        raise ClusterError("The selected partition name is not safe.")
    log_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "slurm_logs")
    result_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "slurm_results")
    gpu_line = f"#SBATCH --gres=gpu:{request.gpu_count}\n" if request.gpu_count else ""
    return f"""#!/usr/bin/env bash
# Generated by SPACEMAP. This fixed probe does not run a user-supplied command.
#SBATCH --job-name={SAFE_JOB_NAME}
#SBATCH --partition={partition.name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={request.cpus}
#SBATCH --mem={request.memory_gb}G
#SBATCH --time={request.hours:02d}:00:00
{gpu_line}#SBATCH --output={log_dir}/{SAFE_JOB_NAME}-%j.out
#SBATCH --error={log_dir}/{SAFE_JOB_NAME}-%j.err
#SBATCH --chdir={result_dir}

set -eu
umask 077
job_id="${{SLURM_JOB_ID:-manual}}"
output_dir="{result_dir}/${{job_id}}"
mkdir -p "${{output_dir}}"
report="${{output_dir}}/resource_report.txt"
{{
    echo "SPACEMAP cluster resource probe"
    date --iso-8601=seconds
    echo "job_id=${{job_id}}"
    echo "host=${{SLURMD_NODENAME:-unknown}}"
    echo "partition=${{SLURM_JOB_PARTITION:-unknown}}"
    echo "cpus=${{SLURM_CPUS_PER_TASK:-unknown}}"
    echo "memory_request={request.memory_gb}G"
    echo "gpu_request={request.gpu_count}"
    uname -a
    free -h
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        echo "nvidia-smi is not available on this node"
    fi
}} > "${{report}}"
echo "Resource report saved to ${{report}}"
"""


def create_probe_script(request: ResourceRequest, partitions: list[PartitionInfo]) -> tuple[Path, PartitionInfo, str]:
    selected, reason = recommend_partition(request, partitions)
    script_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "slurm_scripts")
    script_dir.mkdir(parents=True, exist_ok=True)
    (ASSISTANT_DATA_DIR / "slurm_logs").mkdir(parents=True, exist_ok=True)
    (ASSISTANT_DATA_DIR / "slurm_results").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    script_path = require_assistant_write_path(script_dir / f"resource_probe_{stamp}.sh")
    script_path.write_text(_safe_probe_script(request, selected), encoding="utf-8")
    script_path.chmod(0o700)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": asdict(request.validated()),
        "selected_partition": asdict(selected),
        "reason": reason,
    }
    script_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return script_path, selected, reason


def validate_probe_script(script_path: str | Path) -> str:
    path = require_assistant_write_path(script_path)
    expected_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "slurm_scripts")
    try:
        path.relative_to(expected_dir)
    except ValueError as error:
        raise ClusterError("Only SPACEMAP-generated Slurm scripts can be validated.") from error
    text = path.read_text(encoding="utf-8")
    required = [
        "# Generated by SPACEMAP.",
        f"#SBATCH --job-name={SAFE_JOB_NAME}",
        "SPACEMAP cluster resource probe",
    ]
    if not all(marker in text for marker in required):
        raise ClusterError("The probe script failed the SPACEMAP template check.")
    try:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        request = ResourceRequest(**metadata["request"]).validated()
        partition_payload = dict(metadata["selected_partition"])
        partition_payload["states"] = tuple(partition_payload["states"])
        partition = PartitionInfo(**partition_payload)
        expected = _safe_probe_script(request, partition)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ClusterError("The probe metadata is missing or invalid.") from error
    if text != expected:
        raise ClusterError("The probe script changed after it was generated. Create a new probe instead.")
    result = _run(["bash", "-n", str(path)])
    return "$ bash -n " + str(path) + "\nSyntax and safe-template checks passed." + result.stdout


def submit_probe_script(script_path: str | Path) -> tuple[str, str]:
    """Submit only a generated probe. The GUI must obtain explicit confirmation."""
    validation = validate_probe_script(script_path)
    path = require_assistant_write_path(script_path)
    result = _run(["sbatch", "--parsable", str(path)])
    raw_job_id = result.stdout.strip().split(";", 1)[0]
    if not raw_job_id.isdigit():
        raise ClusterError("Slurm did not return a valid job number.")
    terminal = validation + "\n$ sbatch --parsable " + str(path) + "\n" + result.stdout.strip()
    return raw_job_id, terminal


def cluster_available() -> bool:
    """True when Slurm client commands are present on this machine."""
    return bool(shutil.which("sinfo") and shutil.which("sbatch"))


def cluster_enabled() -> bool:
    """True when the user has switched cluster submission on and Slurm exists."""
    return bool(load_settings().get("cluster", {}).get("enabled")) and cluster_available()
