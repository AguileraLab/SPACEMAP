"""Batch throughput policy and retry behaviour without requiring a GPU."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from spacemap.segmentation_execution import batch_candidates, fastest_batch


def test_candidates_follow_memory_and_block_size():
    assert batch_candidates(0, 224) == [1]
    assert max(batch_candidates(100 * 1024**3, 224)) == 256
    assert max(batch_candidates(8 * 1024**3, 224)) == 32
    assert max(batch_candidates(8 * 1024**3, 448)) == 8


def test_selects_throughput_not_the_largest_allocation():
    times = {1: 1., 2: 1.2, 4: 2.4, 8: 5.}
    assert fastest_batch(list(times), times.__getitem__, lambda e: False) == 2


def test_gpu_memory_failure_retains_the_last_successful_batch():
    seen = []
    def measure(batch):
        seen.append(batch)
        if batch >= 8:
            raise RuntimeError('CUDA out of memory')
        return 1.
    assert fastest_batch([1, 2, 4, 8, 16], measure, lambda e: 'out of memory' in str(e)) == 4
    assert 16 not in seen


def test_does_not_hide_non_memory_errors():
    def broken(batch):
        raise RuntimeError('invalid network')
    with pytest.raises(RuntimeError, match='invalid network'):
        fastest_batch([1, 2], broken, lambda e: False)


def test_time_budget_checked_between_probes():
    now = [0.]
    def measure(batch):
        now[0] += 1
        return 1.
    assert fastest_batch([1, 2, 4, 8], measure, lambda e: False, seconds=4, clock=lambda: now[0]) == 1


@pytest.fixture
def segmenter(monkeypatch):
    from spacemap import segmentation_cellpose as mod
    monkeypatch.setattr(mod, 'CELLPOSE_AVAILABLE', True)
    monkeypatch.setattr(mod, 'GPU_AVAILABLE', True)
    monkeypatch.setattr(mod.CellposeSegmenter, '_load_model', lambda self: None)
    config = mod.SegmentationConfig()
    config.use_gpu = True
    config.gpu_device = 2
    config.flow_threshold = 0
    runner = mod.CellposeSegmenter(config)
    calls, clears, queried = [], [], []
    def memory(device):
        queried.append(device)
        return 80 * 1024**3, 100 * 1024**3
    monkeypatch.setattr(mod.torch.cuda, 'mem_get_info', memory)
    monkeypatch.setattr(mod.torch.cuda, 'memory_reserved', lambda device: 4 * 1024**3)
    monkeypatch.setattr(mod.torch.cuda, 'memory_allocated', lambda device: 1 * 1024**3)
    monkeypatch.setattr(mod.torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(mod.torch.cuda, 'empty_cache', lambda: clears.append(True))
    monkeypatch.setattr(runner, '_auto_batch', lambda bsize: 64)
    mask = np.array([[0, 1], [2, 2]], dtype=np.int32)
    def evaluate(image, *, diameter, flow_threshold, channels, tile_overlap, batch_size, bsize):
        calls.append((batch_size, bsize, diameter, channels, tile_overlap, image.copy()))
        if batch_size > 16:
            raise RuntimeError('CUDA out of memory')
        return mask.copy(), [], None
    runner.model = SimpleNamespace(eval=evaluate)
    return mod, runner, mask, calls, clears, queried


def test_oom_batch_limit_is_reused_without_changing_pixels_or_segmentation_settings(segmenter):
    mod, runner, mask, calls, clears, queried = segmenter
    pixels = np.arange(12).reshape(2, 2, 3)
    first = runner.segment(pixels)[0]
    second = runner.segment(pixels)[0]
    np.testing.assert_array_equal(first, mask)
    np.testing.assert_array_equal(second, mask)
    assert [row[0] for row in calls] == [64, 32, 16, 16]
    assert all(row[1:5] == (224, 0., [1, 3], .1) for row in calls)
    assert all(np.array_equal(row[-1], pixels) for row in calls)
    assert len(clears) == 2  # only failed allocations, never successful tiles
    assert queried and set(queried) == {2}
    assert runner._available_memory() == 83 * 1024**3


def test_manual_mode_and_cpu_fallback_do_not_tune(segmenter, monkeypatch):
    mod, runner, mask, calls, clears, queried = segmenter
    def unwanted(*args):
        raise AssertionError('Should not tune')
    monkeypatch.setattr(runner, '_auto_batch', unwanted)
    runner.config.cellpose_batch_auto = False
    runner.config.cellpose_batch_size = 4
    runner.segment(np.zeros((2, 2, 3)))
    assert calls[-1][0] == 4 and not clears
    queried.clear()
    monkeypatch.setattr(mod, 'GPU_AVAILABLE', False)
    runner.config.cellpose_batch_auto = True
    runner.segment(np.zeros((2, 2, 3)))
    assert calls[-1][0] == 4 and not queried


def test_studio_execution_settings_validate_and_reach_engine(tmp_path):
    from spacemap_gui.studio import segmentation
    changes = {'tab4_cellpose_batch_auto': False, 'tab4_cellpose_batch_size': 256, 'tab4_gpu_device': 2}
    segmentation.check(changes)
    config = segmentation.config_for({**segmentation.values(tmp_path, {}), **changes})
    assert config.cellpose_batch_size == 256 and not config.cellpose_batch_auto and config.gpu_device == 2
    for key, value in [('tab4_cellpose_batch_size', 257), ('tab4_gpu_device', -1), ('tab4_gpu_device', 1.5), ('tab4_cellpose_batch_auto', 'true')]:
        with pytest.raises(ValueError):
            segmentation.check({key: value})
