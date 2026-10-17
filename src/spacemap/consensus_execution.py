"""Execution controls for consensus forests; no changes to voting or features."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import threading
import time


def _cgroup_paths(controller):
    """This process's cgroup and parents, including nested scheduler limits."""
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        return
    for line in memberships:
        _, controllers, relative = line.split(':', 2)
        if controllers and controller not in controllers.split(','):
            continue
        roots = [Path('/sys/fs/cgroup')] if not controllers else [Path('/sys/fs/cgroup') / controller]
        if controllers and controller == 'cpu':
            roots.extend([Path('/sys/fs/cgroup/cpu,cpuacct'), Path('/sys/fs/cgroup/cpuacct,cpu')])
        for root in roots:
            current = root / relative.lstrip('/')
            # Also inspect the mount root in cgroup namespaces.
            paths = [current, *[p for p in current.parents if p == root or root in p.parents]]
            for path in paths:
                yield path, bool(controllers)


def _quota_limits():
    limits = []
    for path, legacy in _cgroup_paths('cpu'):
        try:
            if legacy:
                quota = int((path / 'cpu.cfs_quota_us').read_text())
                period = int((path / 'cpu.cfs_period_us').read_text())
            else:
                q, p = (path / 'cpu.max').read_text().split()
                if q == 'max':
                    continue
                quota, period = int(q), int(p)
            if quota > 0 and period > 0:
                limits.append(max(1, math.floor(quota / period)))
        except (OSError, ValueError):
            pass
    return limits


def _memory_headroom():
    """Available host memory bounded by this job's remaining cgroup allowance."""
    limits = []
    try:
        import psutil
        limits.append(psutil.virtual_memory().available)
    except (ImportError, OSError):
        pass
    for path, legacy in _cgroup_paths('memory'):
        try:
            maximum = (path / ('memory.limit_in_bytes' if legacy else 'memory.max')).read_text().strip()
            if maximum == 'max':
                continue
            used = int((path / ('memory.usage_in_bytes' if legacy else 'memory.current')).read_text())
            limits.append(max(0, int(maximum) - used))
        except (OSError, ValueError):
            pass
    return min(limits) if limits else None


def cpu_budget(requested=0):
    """Auto adapts to CPU and memory headroom; explicit counts respect allocation.

    Ignore LOKY_MAX_CPU_COUNT here: Studio's old two-core default is exactly the
    limit this user setting replaces. Native numerical thread pools stay at one.
    """
    if (isinstance(requested, bool) or not isinstance(requested, (int, float))
            or not 0 <= requested <= 4096 or int(requested) != requested):
        raise ValueError('CPU cores must be a whole number from 0 (Auto) to 4096.')
    requested = int(requested)
    limits = [os.cpu_count() or 1, *_quota_limits()]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for key in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        match = re.match(r'^(\d+)', os.environ.get(key, ''))
        if match and int(match[1]) > 0:
            limits.append(int(match[1]))
    available = max(1, min(limits))
    memory = _memory_headroom()
    # A conservative workspace allowance, not a claim that forest memory is
    # bounded: the data and completed trees also occupy memory. Keep eight as
    # the fallback only when headroom cannot be measured.
    auto_workers = min(32, available, max(1, memory // (4 * 2**30)) if memory is not None else 8)
    return {'requested': requested, 'available': available,
            'workers': min(requested or auto_workers, available), 'auto_workers': auto_workers,
            'memory_available_gib': round(memory / 2**30, 1) if memory is not None else None}


def worker_environment(config):
    budget = cpu_budget(config.get('cpu_workers', 0))
    return {**{name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
            'LOKY_MAX_CPU_COUNT': str(budget['workers'])}


def gpu_probe():
    """Run in the worker or a short-lived probe, never initialise CUDA in the UI."""
    try:
        import cuml
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
        from packaging.version import Version

        if Version(cuml.__version__) < Version('26.8'):
            raise ValueError('cuML 26.08 or newer is required for balanced classes and unlimited tree depth.')
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise ValueError('No accessible CUDA GPU.')
        # Constructors must support the actual model specification; no silent
        # unweighted or depth-limited substitute on older installations.
        RandomForestClassifier(class_weight='balanced', max_depth=None, output_type='numpy')
        device = cp.cuda.Device()
        value = cp.ones(1, dtype=cp.float32)
        if float(value.sum().get()) != 1:
            raise ValueError('CUDA computation check failed.')
        free, total = device.mem_info
        name = cp.cuda.runtime.getDeviceProperties(device.id)['name']
        return {'available': True, 'version': cuml.__version__, 'device': device.id,
                'name': name.decode() if isinstance(name, bytes) else str(name),
                'free_bytes': int(free), 'total_bytes': int(total)}
    except Exception as exc:
        return {'available': False, 'reason': f'GPU unavailable: {exc}'}


def resolve_execution(backend='cpu', cpu_workers=0, gpu_bins=256):
    if backend not in ('cpu', 'gpu', 'auto'):
        raise ValueError('Choose CPU, GPU, or GPU if available.')
    if (isinstance(gpu_bins, bool) or not isinstance(gpu_bins, (int, float))
            or not 16 <= gpu_bins <= 4096 or int(gpu_bins) != gpu_bins):
        raise ValueError('GPU split bins must be a whole number from 16 to 4096.')
    gpu_bins = int(gpu_bins)
    info = {**cpu_budget(cpu_workers), 'requested_backend': backend, 'backend': 'cpu',
            'gpu_bins': gpu_bins}
    if backend != 'cpu':
        info['gpu'] = gpu_probe()
        if info['gpu']['available']:
            info['backend'] = 'gpu'
        elif backend == 'gpu':
            raise ValueError(info['gpu']['reason'] + ' Choose CPU or configure a compatible GPU environment.')
        else:
            info['fallback_reason'] = info['gpu']['reason']
    return info


class GPUForest:
    """Native cuML RF with host-array outputs and the same label-encoder contract.

    GPU split quantiles differ from sklearn; this is explicitly opt-in. No CPU
    fallback occurs after fitting begins, including on CUDA out-of-memory.
    """
    def __init__(self, n_estimators, random_state, n_bins=256):
        from cuml.ensemble import RandomForestClassifier
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_bins=n_bins,
            class_weight='balanced', max_depth=None, max_features='sqrt',
            bootstrap=True, max_samples=1.0, min_samples_leaf=1, min_samples_split=2,
            split_criterion='gini', n_streams=1, output_type='numpy')

    def fit(self, X, y):
        import numpy as np
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32))
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        return self

    def predict_proba(self, X):
        import numpy as np
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=np.float32)))

    def predict(self, X):
        return self.classes_.take(self.predict_proba(X).argmax(axis=1))


@contextmanager
def operation_progress(notify, description, interval=None):
    """Report stage boundaries; periodic messages are an explicit caller opt-in."""
    start = time.monotonic()
    stopped = threading.Event()
    errors = []
    thread = None
    def heartbeat():
        while not stopped.wait(interval):
            try:
                notify(f'{description} · {time.monotonic() - start:.0f}s elapsed')
            except BaseException as exc:
                errors.append(exc)
                return
    if notify:
        notify(description)
    if notify and interval is not None:
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    try:
        yield
        if errors:
            raise errors[0]
    finally:
        stopped.set()
        if thread is not None:
            thread.join()
    if notify:
        notify(f'{description} · finished in {time.monotonic() - start:.1f}s')


if __name__ == '__main__':
    print(json.dumps(gpu_probe()))
