"""Resource selection, CPU parity, and optional GPU failure/label semantics."""

import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from spacemap import consensus_execution as execution
from spacemap.ml_consensus import (_create_classifier, augment_unreachable_classes,
                                   predict_non_consensus, train_and_validate, train_final_model)


def test_cpu_budget_respects_affinity_slurm_and_quota_not_legacy_loky(monkeypatch):
    monkeypatch.setattr(execution.os, 'cpu_count', lambda: 256)
    monkeypatch.setattr(execution.os, 'sched_getaffinity', lambda pid: set(range(64)))
    monkeypatch.setattr(execution, '_quota_limits', lambda: [24])
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '128')
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    monkeypatch.setattr(execution, '_memory_headroom', lambda: 128 * 2**30)
    assert execution.cpu_budget(32) == {'requested': 32, 'available': 16, 'workers': 16,
                                       'auto_workers': 16, 'memory_available_gib': 128.0}
    assert execution.cpu_budget(0)['workers'] == 16
    assert execution.cpu_budget(3)['workers'] == 3
    env = execution.worker_environment({'cpu_workers': 12})
    assert env['LOKY_MAX_CPU_COUNT'] == '12'
    assert env['OPENBLAS_NUM_THREADS'] == env['OMP_NUM_THREADS'] == '1'
    for bad in (-1, 1.5, True, 9999):
        with pytest.raises(ValueError):
            execution.cpu_budget(bad)


def test_auto_scales_on_large_hosts_but_respects_memory(monkeypatch):
    monkeypatch.setattr(execution.os, 'cpu_count', lambda: 256)
    monkeypatch.setattr(execution.os, 'sched_getaffinity', lambda pid: set(range(256)))
    monkeypatch.setattr(execution, '_quota_limits', lambda: [])
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '256')
    for gib, expected in [(450, 32), (64, 16), (16, 4), (0, 1)]:
        monkeypatch.setattr(execution, '_memory_headroom', lambda: gib * 2**30)
        assert execution.cpu_budget()['workers'] == expected
        assert execution.cpu_budget(48)['workers'] == 48  # explicit setting is not Auto
    monkeypatch.setattr(execution, '_memory_headroom', lambda: None)
    assert execution.cpu_budget()['workers'] == 8


def test_memory_uses_remaining_nested_cgroup_allowance(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=450 * 2**30))
    texts = {'/proc/self/cgroup': '0::/slurm/task\n',
             '/sys/fs/cgroup/slurm/task/memory.max': 'max',
             '/sys/fs/cgroup/slurm/memory.max': str(32 * 2**30),
             '/sys/fs/cgroup/slurm/memory.current': str(8 * 2**30),
             '/sys/fs/cgroup/memory.max': 'max'}
    def read(path, *args, **kwargs):
        if str(path) not in texts:
            raise FileNotFoundError(str(path))
        return texts[str(path)]
    monkeypatch.setattr(execution.Path, 'read_text', read)
    assert execution._memory_headroom() == 24 * 2**30


def test_nested_cgroup_quota(monkeypatch):
    texts = {'/proc/self/cgroup': '0::/slurm/task\n',
             '/sys/fs/cgroup/slurm/task/cpu.max': 'max 100000',
             '/sys/fs/cgroup/slurm/cpu.max': '650000 100000',
             '/sys/fs/cgroup/cpu.max': 'max 100000'}
    def read(path, *args, **kwargs):
        if str(path) not in texts:
            raise FileNotFoundError(str(path))
        return texts[str(path)]
    monkeypatch.setattr(execution.Path, 'read_text', read)
    assert execution._quota_limits() == [6]


def test_gpu_unavailability_is_explicit_and_auto_is_recorded(monkeypatch):
    monkeypatch.setattr(execution, 'gpu_probe', lambda: {'available': False, 'reason': 'cuML missing'})
    with pytest.raises(ValueError, match='cuML missing'):
        execution.resolve_execution('gpu')
    found = execution.resolve_execution('auto')
    assert found['backend'] == 'cpu' and found['fallback_reason'] == 'cuML missing'
    monkeypatch.setattr(execution, 'gpu_probe', lambda: pytest.fail('CPU must not initialise CUDA'))
    assert execution.resolve_execution('cpu')['backend'] == 'cpu'


def test_gpu_adapter_retains_balance_trees_and_decodes_classes(monkeypatch):
    seen = {}
    class NativeForest:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def fit(self, X, y):
            assert X.dtype == np.float32 and y.dtype == np.int32
            self.classes_ = np.array([1, 3])
            return self
        def predict_proba(self, X):
            return np.array([[.1, .9], [.5, .5]])
    monkeypatch.setitem(sys.modules, 'cuml.ensemble', SimpleNamespace(RandomForestClassifier=NativeForest))
    model = _create_classifier('random_forest', 1500, 42, execution_backend='gpu', gpu_bins=512)
    model.fit(np.ones((2, 2)), np.array([1, 3]))
    assert seen['n_estimators'] == 1500 and seen['class_weight'] == 'balanced'
    assert seen['max_depth'] is None and seen['n_bins'] == 512 and seen['n_streams'] == 1
    assert model.predict(np.ones((2, 2))).tolist() == [3, 1]
    with pytest.raises(ValueError, match='Random Forest only'):
        _create_classifier('svm', 1500, 42, execution_backend='gpu')


def test_cpu_parallelism_preserves_trees_split_metrics_and_single_pass(monkeypatch):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(350, 8)))
    labels = pd.Series(np.where(frame[0] > .6, 'T', np.where(frame[1] > 0, 'B', 'Other')))
    low = train_and_validate(frame, labels, .3, 60, 42, cpu_workers=1)
    high = train_and_validate(frame, labels, .3, 60, 42, cpu_workers=4)
    assert low[1] == high[1]
    np.testing.assert_array_equal(low[2], high[2])  # identical held-out population
    np.testing.assert_array_equal(low[3], high[3])
    for a, b in zip(low[0].estimators_, high[0].estimators_):
        np.testing.assert_array_equal(a.tree_.__getstate__()['nodes'], b.tree_.__getstate__()['nodes'])
        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
    encoder = LabelEncoder().fit(labels)
    scaler = StandardScaler().fit(frame)
    final = train_final_model(frame, labels, encoder, scaler, 60, 42, cpu_workers=4)
    transformed = scaler.transform(frame)
    expected = encoder.inverse_transform(final.predict(transformed))
    probability = final.predict_proba(transformed).max(axis=1)
    calls = []
    original = final.predict_proba
    monkeypatch.setattr(final, 'predict', lambda X: pytest.fail('RF must not traverse twice'))
    def proba(X):
        calls.append(len(X))
        return original(X)
    monkeypatch.setattr(final, 'predict_proba', proba)
    predicted, confidence = predict_non_consensus(final, scaler, frame, encoder)
    assert calls == [len(frame)]
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_allclose(confidence, probability, atol=1e-12)


def test_non_rf_keeps_independent_decision_labels():
    model = SimpleNamespace(predict=lambda X: np.array([1]),
                            predict_proba=lambda X: np.array([[.9, .1]]))
    frame = pd.DataFrame([[1.]])
    scaler = StandardScaler().fit(frame)
    encoder = LabelEncoder().fit(['A', 'B'])
    labels, confidence = predict_non_consensus(model, scaler, frame, encoder)
    assert labels.tolist() == ['B'] and confidence.tolist() == [.9]


def test_reference_only_rare_class_cannot_be_rescued():
    frame = pd.DataFrame({'consensus_phenotype': ['A', 'B', None, None],
                          'majority_phenotype': ['A', 'B', 'C', 'D'],
                          'consensus_count': [3, 3, 2, 2]})
    mask, labels, rescued = augment_unreachable_classes(
        frame, [True, True, False, False], 3, eligible_mask=[True, True, False, True])
    assert mask.tolist() == [True, True, False, True]
    assert labels.tolist() == ['A', 'B', 'D'] and rescued == {'D': 1}


def test_default_progress_reports_only_stage_boundaries(monkeypatch):
    messages = []
    now = [10.0]
    monkeypatch.setattr(execution.time, 'monotonic', lambda: now[0])
    def unexpected_thread(*args, **kwargs):
        pytest.fail('Default progress must not start a heartbeat thread')
    monkeypatch.setattr(execution.threading, 'Thread', unexpected_thread)
    with execution.operation_progress(messages.append, 'Training'):
        now[0] += 65.5
    assert messages == ['Training', 'Training · finished in 65.5s']


def test_heartbeat_reports_elapsed_and_stops():
    messages = []
    with execution.operation_progress(messages.append, 'Training', interval=.01):
        time.sleep(.05)
    assert any('elapsed' in line for line in messages) and 'finished' in messages[-1]
    count = len(messages)
    time.sleep(.03)
    assert len(messages) == count


def test_heartbeat_propagates_compute_cancellation():
    from spacemap_gui.studio.job_stop import ComputationCancelled
    def report(message):
        if 'elapsed' in message:
            raise ComputationCancelled('Stop')
    with pytest.raises(ComputationCancelled):
        with execution.operation_progress(report, 'Training', interval=.01):
            time.sleep(.05)


def test_native_gpu_when_installed():
    info = execution.gpu_probe()
    if not info['available']:
        pytest.skip(info['reason'])
    model = execution.GPUForest(20, 42)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > .9
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1, atol=1e-5)
