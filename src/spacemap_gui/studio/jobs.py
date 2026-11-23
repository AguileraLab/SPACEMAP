"""Durable Studio jobs. Only registered projects and fixed workers may run."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from spacemap_workflows.leiden_job import write_json
from spacemap.sharing import open_lock
from . import job_stop
from .job_stop import Cancelled

ACTIVE = {"starting", "running", "stopping"}
KINDS = {
    "resolve_all",
    "mr_ml",
    "train_reference_model",
    "reference_collection",
    "model_review",
    "marker_figures",
    "phenotyping_images",
    "subtract_background",
    "extract_rois",
    "detect_roi_cores",
    "marker_regions",
    "classify_images",
    "export_annotations",
    "import_annotations",
    "segment_images",
    "register_pairs",
    "detect_tissue",
    "clean_tissue",
    "detect_artifacts",
    "train_artifact_model",
    "artifact_batch",
    "subtract_annotations",
    "classify_shared",
    "remove_small_pieces",
    "spatial_density",
    "tls_candidates",
    "gland_types",
    "spatial_triads",
    "spatial_screening",
    "spatial_umap",
    "save_class_polygons",
    "subtract_artifacts",
    "export_classes",
    "apply_pieces",
    "make_zones",
    "leiden",
    "name_clusters",
    "neighbourhoods",
    "train_cellpose",
    "score_cellpose",
    "correct_batches",
    "som",
    "vortex",
    "export_table",
    "write_annotations",
    "consensus_ml",
    "subphenotype",
    "cell_interactions",
    "cell_tagging",
    "remove_background",
    "export_background",
    "detect_background",
}


def root(project):
    return Path(project).resolve() / "outputs" / "studio" / "jobs"


def directory(project, job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Choose a saved job.")
    folder = root(project) / job_id
    if not folder.is_dir():
        raise ValueError("This job was not found in the current project.")
    return folder


def status(project, job_id):
    folder = directory(project, job_id)
    result = json.loads((folder / "status.json").read_text())
    result = job_stop.reconcile(folder, result)
    if result["status"] in ACTIVE and (folder / "cancel").exists():
        record = folder / "process.json"
        process = json.loads(record.read_text()) if record.exists() else {}
        result.update(status="stopping", message=job_stop.message(folder, process))
    if result["status"] == "done":
        from .cluster_profiles import for_job
        result = for_job(project, folder, result)
    result['hidden'] = (root(project) / 'hidden' / job_id).exists()
    try:
        request = json.loads((folder / 'request.json').read_text())
        result['summary'] = describe(request.get('kind', result.get('kind')), request.get('settings', {}))
        if result.get('kind') == 'train_cellpose':
            training = request.get('settings', {}).get('training_session') or {}
            name = (result.get('result') or {}).get('name')
            if not name and training.get('name'):
                name = f"{training['name']}_v{len(training.get('versions', [])) + 1}"
            if name:
                result['model_name'] = name
    except (OSError, ValueError, TypeError):
        result['summary'] = ''
    return result


def describe(kind, settings):
    """Short, factual scope from the saved request, including older jobs."""
    if kind == 'phenotyping_images':
        work = settings.get('work') or []
        if work:
            first = work[0]
            merged = dict(first.get('settings', {}))
            merged['sources'] = [item.get('settings', {}) for item in work]
            merged.pop('selected_images', None)
            return first.get('kind', '').replace('_', ' ').upper() + ' · ' + describe(first.get('kind'), merged)
    parts = []
    sources = settings.get('sources') or [settings]
    images = [(source.get('table'), image) for source in sources for image in (source.get('selected_images') or [])]
    if images:
        count = len(set(images))
        parts.append(str(images[0][1]) if count == 1 else f'{count} images')
    elif settings.get('roi'):
        parts.append(str(settings['roi']))
    elif settings.get('table'):
        parts.append(Path(settings['table']).name)
    reference = settings.get('training_column') or settings.get('reference_column')
    output = settings.get('output_col') or settings.get('output_column')
    if kind in ('train_reference_model', 'reference_collection') and settings.get('action', 'train') == 'train':
        parts.append(('Train from ' + reference) if reference else 'Train reference model')
    elif output:
        parts.append((reference + ' → ' if reference else 'Output: ') + output)
    elif kind == 'export_table':
        parts.append('Export CSV')
    features = settings.get('markers')
    if isinstance(features, list) and features:
        parts.append(f'{len(features)} feature' + ('s' if len(features) != 1 else ''))
    return ' · '.join(parts)


def clear_history(project, job_ids=None):
    """Hide finished cards only. Preserve requests, logs, models and Parquet outputs."""
    path = root(project)
    if job_ids is None:
        folders = [folder for folder in path.iterdir() if folder.is_dir() and re.fullmatch(r'[0-9a-f]{32}', folder.name)] if path.exists() else []
    else:
        if not isinstance(job_ids, list) or not job_ids or any(not isinstance(item, str) for item in job_ids):
            raise ValueError('Select finished jobs to clear.')
        folders = [directory(project, job_id) for job_id in dict.fromkeys(job_ids)]
    cleared, active = [], []
    for folder in folders:
        if not (folder / 'status.json').is_file():
            continue
        current = json.loads((folder / 'status.json').read_text())
        if current.get('status') in ACTIVE:
            active.append(folder.name)
            continue
        hidden = path / 'hidden'
        hidden.mkdir(exist_ok=True)
        (hidden / folder.name).touch()
        cleared.append(folder.name)
    return {'cleared': cleared, 'active': active}


def _recent_folders(project, limit):
    from itertools import groupby

    path = root(project)
    if not path.exists():
        return []
    def submitted(folder):
        # The request is immutable after submission. The directory's mtime
        # changes when results/profile caches are created or a stop is requested,
        # which can incorrectly put an older analysis ahead of its later labels.
        try:
            return (folder / "request.json").stat().st_mtime_ns
        except FileNotFoundError:
            return folder.stat().st_mtime_ns

    entries = [(submitted(folder), folder) for folder in path.iterdir()
               if folder.is_dir() and re.fullmatch(r"[0-9a-f]{32}", folder.name)]
    entries.sort(key=lambda item: item[0], reverse=True)

    def creation_time(folder):
        # Some filesystems give rapidly submitted jobs the same mtime. Consult
        # the recorded timestamp only for ties, avoiding reads of large result
        # manifests during normal polling. Legacy records have no finer time.
        try:
            stamp = json.loads((folder / "status.json").read_text()).get("started", "")
            when = datetime.fromisoformat(stamp).timestamp() if stamp else 0
        except (OSError, ValueError, TypeError):
            when = 0
        return when, folder.name

    ordered = []
    for _, group in groupby(entries, key=lambda item: item[0]):
        folders = [item[1] for item in group]
        if len(folders) > 1:
            folders.sort(key=creation_time, reverse=True)
        ordered.extend(folders)
    # A job being started has its folder a moment before its status file.
    return [folder for folder in ordered[:limit] if (folder / "status.json").exists()]


def history(project, limit=300):
    """Read persisted records lazily, without preparing profiles or reconciling workers.

    Name/output provenance only needs saved records. Stop reading as soon as the
    caller finds its run, instead of enriching hundreds of unrelated results.
    """
    for folder in _recent_folders(project, limit):
        yield json.loads((folder / "status.json").read_text())


def recent(project, limit=20):
    return [status(project, folder.name) for folder in _recent_folders(project, limit)]


def start(project, kind, settings, *, request_id=None):
    import fcntl
    import psutil

    if kind not in KINDS:
        raise ValueError("This background job is not available.")
    from .spatial_sources import OUTPUT_KINDS, output_folder
    if kind in OUTPUT_KINDS:
        settings = {**settings, 'spatial_output': str(output_folder(project))}
    path = root(project)
    path.mkdir(parents=True, exist_ok=True)
    with open_lock(path / ".launch.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = recent(project)
        for job in previous:
            if request_id and job.get("request_id") == request_id:
                return job
        active = next((job for job in previous if job["status"] in ACTIVE), None)
        if active is not None:
            try:
                host = json.loads((path / active["id"] / "process.json").read_text()).get("host")
            except (OSError, ValueError, KeyError):
                host = None
            where = f" on {host}" if host else ""
            raise ValueError(
                f"A Studio job is already running for this project{where}. Wait or stop it in Jobs. "
                "This lock is shared across nodes. To run separately, open an independent project folder "
                "with its own outputs; changing only the output folder does not remove the lock."
            )
        execution_env = {}
        if kind in ("consensus_ml", "mr_ml"):
            from spacemap.consensus_execution import worker_environment
            # Validate before creating an active job record.
            execution_env = worker_environment(settings.get("config", {}))
        job_id = uuid.uuid4().hex
        folder = path / job_id
        folder.mkdir()
        now = datetime.now(timezone.utc).isoformat()
        write_json(
            folder / "request.json", {"project": str(Path(project).resolve()), "kind": kind, "settings": settings}
        )
        write_json(
            folder / "status.json",
            {
                "id": job_id,
                "kind": kind,
                "request_id": request_id,
                "status": "starting",
                "progress": 0,
                "message": "Starting…",
                "started": now,
            },
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        env.update(
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2",
            LOKY_MAX_CPU_COUNT="2",
        )
        env.update(execution_env)
        with (folder / "terminal.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "spacemap_gui.studio.jobs", str(folder)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
        write_json(
            folder / "process.json",
            {"pid": child.pid, "created": psutil.Process(child.pid).create_time(), "host": socket.gethostname(),
             "stop_protocol": job_stop.PROTOCOL if kind in job_stop.INTERRUPTIBLE else 0},
        )
        threading.Thread(target=child.wait, daemon=True).start()
        if kind in job_stop.INTERRUPTIBLE:
            # Separate from both the UI server and scientific worker. A native
            # fit cannot block it, and reconnecting from another host still works.
            try:
                with (folder / "terminal.log").open("a") as log:
                    watcher = subprocess.Popen(
                        [sys.executable, "-m", "spacemap_gui.studio.job_stop", str(folder)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                        start_new_session=True,
                    )
                threading.Thread(target=watcher.wait, daemon=True).start()
            except OSError:
                record = json.loads((folder / "process.json").read_text())
                record["stop_protocol"] = 0
                write_json(folder / "process.json", record)
        return status(project, job_id)


def cancel(project, job_id):
    folder = directory(project, job_id)
    current = status(project, job_id)
    if current["status"] not in ACTIVE:
        return current
    # The independent watcher interrupts audited compute jobs; other jobs keep
    # their safe checkpoints until their output transactions are audited too.
    (folder / "cancel").touch()
    return status(project, job_id)


def log(project, job_id):
    path = directory(project, job_id) / "terminal.log"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        end = handle.seek(0, 2)
        handle.seek(max(0, end - 64000))
        return handle.read().decode(errors="replace")


def worker(folder):
    from spacemap_gui.run_log import ResourceMonitor, record_run

    folder = Path(folder)
    request = json.loads((folder / "request.json").read_text())
    current = json.loads((folder / "status.json").read_text())
    monitor = ResourceMonitor().start()
    output_guard = job_stop.OutputGuard(folder)

    def report(message, progress=None):
        if (folder / "cancel").exists() and output_guard.handle is None:
            exception = job_stop.ComputationCancelled if request["kind"] in job_stop.INTERRUPTIBLE else Cancelled
            raise exception("Stopped by the user.")
        current.update(status="running", message=str(message))
        if progress is not None:
            current["progress"] = min(1, max(0, float(progress)))
        write_json(folder / "status.json", current)
        print(message, flush=True)

    report.protect_output = output_guard.protect
    report.finish_output = output_guard.close

    try:
        report("Preparing the selected data…", 0)
        from . import tasks

        if request["kind"] not in KINDS:
            raise ValueError("Unknown job type.")
        result = getattr(tasks, request["kind"])(request["settings"], folder, report)
        output_guard.protect(allow_cancel=request["kind"] not in job_stop.INTERRUPTIBLE)
        from . import spatial_exports
        try:
            result = spatial_exports.publish(request['project'], request['kind'], request['settings'], folder, result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Analysis and cell-table publication already succeeded. Keep that
            # result usable if a separate export destination becomes unavailable.
            result = {**result, 'export_warning': str(exc)}
            print(f'Spatial export failed; job results are retained: {exc}', flush=True)
        current.update(status="done", progress=1, message="Finished. The result is ready.", result=result)
        from . import annotation_workflow
        if request['kind'] in annotation_workflow.ANNOTATION_JOBS:
            try:
                annotation_workflow.record(request['project'], event={'operation': request['kind'],
                    'job_id': folder.name, 'job_request': str(folder / 'request.json'),
                    'job_result': result, 'status': 'done'})
            except Exception as exc:
                # Geometry/model output already exists; keep success distinct
                # from a catalogue failure and make the latter visible.
                print(f'Workflow record failed: {exc}', flush=True)
                current['workflow_warning'] = str(exc)
                current['message'] = 'Result saved; workflow record needs refreshing from Export.'
    except (Cancelled, job_stop.ComputationCancelled) as exc:
        if request["kind"] in job_stop.INTERRUPTIBLE:
            output_guard.protect(allow_cancel=True, saving=False)
            job_stop.stop_helpers(folder)
        current.update(status="cancelled", message=str(exc))
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        current.update(status="failed", message=str(exc) or type(exc).__name__)
    finally:
        output_guard.protect(allow_cancel=True, saving=False)
        current["usage"] = monitor.stop()
        current["finished"] = datetime.now(timezone.utc).isoformat()
        try:
            record_run(
                request["project"],
                "studio",
                request["kind"],
                started_at=datetime.fromisoformat(current["started"]),
                usage=current["usage"],
                status=current["status"],
                error=current["message"] if current["status"] == "failed" else "",
                settings=request["settings"],
                details={"job_id": folder.name, "result": current.get("result")},
            )
        finally:
            try:
                write_json(folder / "status.json", current)
            finally:
                output_guard.close()
    return 0 if current["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(worker(sys.argv[1]))
