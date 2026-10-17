"""Real isolated workers: interrupt blocked fits, preserve saves, verify identity."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import pytest

from spacemap_gui.studio import jobs, job_stop
from spacemap_workflows.leiden_job import write_json


BOOTSTRAP = r'''
import json, os, signal, subprocess, sys, time, types
from pathlib import Path
log = types.ModuleType('spacemap_gui.run_log')
class Monitor:
    def start(self): return self
    def stop(self): return {'duration_s': 0.1}
log.ResourceMonitor = Monitor
log.record_run = lambda *args, **kwargs: None
sys.modules[log.__name__] = log
tasks = types.ModuleType('spacemap_gui.studio.tasks')
def work(settings, folder, report):
    folder = Path(folder)
    mode = settings.get('mode', 'fit')
    if mode == 'finish': return {'complete': True}
    if mode == 'tag-staging':
        from spacemap_gui.studio.tagging_outputs import StagedTables
        staged = StagedTables(Path(settings['target']).parent, folder)
        (staged.directory / 'partial.csv').write_text('incomplete')
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == 'save':
        report('Saving result', .98)
        report.protect_output()
        target = Path(settings['target'])
        temporary = target.with_suffix('.tmp')
        temporary.write_text('incomplete')
        (folder / 'ready').touch()
        while not (folder / 'release').exists(): time.sleep(.02)
        temporary.write_text('complete result')
        temporary.replace(target)
        report('Saved result', 1.)
        return {'complete': True, 'file': str(target)}
    if mode in {'helper', 'detached'}:
        ready = folder / 'helper-ready'
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; from pathlib import Path; '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            'Path(' + repr(str(ready)) + ').touch(); time.sleep(120)'], start_new_session=mode == 'detached')
        (folder / 'helper.json').write_text(json.dumps({'pid': child.pid}))
        while not ready.exists(): time.sleep(.02)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    report('Fitting a model with no progress callbacks', .4)
    (folder / 'ready').touch()
    while not (folder / 'release').exists(): time.sleep(.02)
    if mode == 'checkpoint':
        report('Next checkpoint', .5)
    else:
        # A broad scientific callback handler must not swallow compute cancellation.
        try: report('Next checkpoint', .5)
        except Exception: pass
    (folder / 'incorrectly-continued').touch()
    return {'complete': True}
tasks.consensus_ml = work
tasks.register_pairs = work
tasks.cell_tagging = work
sys.modules[tasks.__name__] = tasks
'''


def until(check, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = check()
        if found:
            return found
        time.sleep(.03)
    pytest.fail('Worker did not reach the expected state in time')


@pytest.fixture
def launch(tmp_path, monkeypatch):
    bootstrap = tmp_path / 'bootstrap'
    bootstrap.mkdir()
    (bootstrap / 'sitecustomize.py').write_text(BOOTSTRAP)
    monkeypatch.setenv('PYTHONPATH', str(bootstrap) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    started = []

    def start(**settings):
        kind = settings.pop('kind', 'consensus_ml')
        project = tmp_path / 'project'
        project.mkdir(exist_ok=True)
        job = jobs.start(project, kind, settings)
        folder = jobs.directory(project, job['id'])
        started.append((project, folder))
        return project, folder, job

    yield start
    # Failed tests must not leave their synthetic workers/helpers behind.
    for project, folder in started:
        (folder / 'release').touch()
        record = json.loads((folder / 'process.json').read_text())
        if job_stop._alive(record):
            try:
                proc = psutil.Process(record['pid'])
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                proc.wait(5)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize('mode', ['helper', 'detached'])
@pytest.mark.parametrize('kind', ['consensus_ml', 'cell_tagging'])
def test_stop_interrupts_fit_and_ignored_term_helpers_and_unlocks_project(launch, mode, kind):
    project, folder, job = launch(mode=mode, kind=kind)
    until(lambda: (folder / 'ready').exists())
    helper = psutil.Process(json.loads((folder / 'helper.json').read_text())['pid'])
    helper_record = {'pid': helper.pid, 'created': helper.create_time()}
    started = time.monotonic()
    stopped = jobs.cancel(project, job['id'])
    assert time.monotonic() - started < 1
    assert stopped['status'] == 'stopping'
    assert 'Stopping computation' in stopped['message']
    with pytest.raises(ValueError, match='already running'):
        jobs.start(project, 'consensus_ml', {'mode': 'finish'})
    until(lambda: jobs.status(project, job['id'])['status'] == 'cancelled')
    assert time.monotonic() - started < 12
    assert not job_stop._alive(helper_record)
    assert not (folder / 'incorrectly-continued').exists()
    state = json.loads((folder / 'status.json').read_text())
    assert state['status'] == 'cancelled'
    replacement = jobs.start(project, 'consensus_ml', {'mode': 'finish'})
    until(lambda: jobs.status(project, replacement['id'])['status'] == 'done')


@pytest.mark.parametrize('kind', ['consensus_ml', 'cell_tagging'])
def test_stop_during_save_finishes_transaction_and_keeps_result(launch, tmp_path, kind):
    target = tmp_path / 'existing-result.txt'
    target.write_text('original result')
    project, folder, job = launch(mode='save', target=str(target), kind=kind)
    until(lambda: (folder / 'ready').exists())
    assert jobs.cancel(project, job['id'])['message'] == 'Finishing the result save safely…'
    time.sleep(.3)
    assert jobs.status(project, job['id'])['status'] == 'stopping'
    assert target.read_text() == 'original result'
    (folder / 'release').touch()
    result = until(lambda: (state if (state := jobs.status(project, job['id']))['status'] == 'done' else None))
    assert result['result']['complete'] and target.read_text() == 'complete result'
    assert jobs.cancel(project, job['id'])['status'] == 'done'


def test_hard_stop_tagging_removes_partial_staging_and_preserves_output(launch, tmp_path):
    target = tmp_path / 'tagged.csv'
    target.write_text('original')
    project, folder, job = launch(kind='cell_tagging', mode='tag-staging', target=str(target))
    until(lambda: (folder / 'ready').exists())
    staging = Path(json.loads((folder / 'tagging-staging.json').read_text())['directory'])
    assert staging.exists()
    jobs.cancel(project, job['id'])
    until(lambda: jobs.status(project, job['id'])['status'] == 'cancelled')
    until(lambda: not staging.exists())
    assert target.read_text() == 'original'
    assert not (folder / 'incorrectly-continued').exists()


def test_unaudited_output_job_keeps_safe_checkpoint_behavior(launch):
    project, folder, job = launch(kind='register_pairs', mode='checkpoint')
    until(lambda: (folder / 'ready').exists())
    assert json.loads((folder / 'process.json').read_text())['stop_protocol'] == 0
    assert 'safe checkpoint' in jobs.cancel(project, job['id'])['message']
    time.sleep(.25)
    assert jobs.status(project, job['id'])['status'] == 'stopping'
    (folder / 'release').touch()
    until(lambda: jobs.status(project, job['id'])['status'] == 'cancelled')
    assert not (folder / 'incorrectly-continued').exists()


def test_callback_cancellation_also_cleans_up_helpers(launch):
    project, folder, job = launch(mode='helper')
    until(lambda: (folder / 'ready').exists())
    helper = psutil.Process(json.loads((folder / 'helper.json').read_text())['pid'])
    record = {'pid': helper.pid, 'created': helper.create_time()}
    (folder / 'cancel').touch()
    (folder / 'release').touch()
    until(lambda: jobs.status(project, job['id'])['status'] == 'cancelled')
    assert not job_stop._alive(record)
    assert not (folder / 'incorrectly-continued').exists()


def test_pending_stop_cannot_enter_output_transaction(tmp_path):
    (tmp_path / 'cancel').touch()
    guard = job_stop.OutputGuard(tmp_path)
    with pytest.raises(job_stop.Cancelled):
        guard.protect()
    assert guard.handle is None and not (tmp_path / 'saving').exists()


def test_unrelated_process_and_other_host_are_never_signalled(tmp_path):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        record = {'pid': child.pid, 'created': psutil.Process(child.pid).create_time(), 'host': socket.gethostname()}
        with pytest.raises(ValueError, match='identity'):
            job_stop._verified_worker(tmp_path, record)
        with pytest.raises(ValueError, match='host'):
            job_stop._verified_worker(tmp_path, {**record, 'host': 'different-host'})
        with pytest.raises(psutil.NoSuchProcess):
            job_stop._verified_worker(tmp_path, {**record, 'created': record['created'] - 1})
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_stale_pid_reconciles_without_signalling_current_process(tmp_path):
    record = {'pid': os.getpid(), 'created': psutil.Process().create_time() - 1, 'host': socket.gethostname()}
    write_json(tmp_path / 'process.json', record)
    current = {'id': 'a' * 32, 'status': 'running'}
    write_json(tmp_path / 'status.json', current)
    (tmp_path / 'cancel').touch()
    assert job_stop.reconcile(tmp_path, current)['status'] == 'cancelled'
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'cancelled'


def test_parent_exit_does_not_unlock_while_helper_is_alive(tmp_path):
    current = {'id': 'b' * 32, 'status': 'running'}
    write_json(tmp_path / 'status.json', current)
    write_json(tmp_path / 'process.json', {'pid': os.getpid(), 'created': 0, 'host': socket.gethostname()})
    write_json(tmp_path / 'terminating.json', {'processes': [{'pid': os.getpid(), 'created': psutil.Process().create_time()}]})
    (tmp_path / 'cancel').touch()
    assert job_stop.reconcile(tmp_path, current)['status'] == 'running'
