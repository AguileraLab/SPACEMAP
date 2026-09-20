"""Exact CPU/GPU distance parity and safe fallback without changing geometry."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiLineString, Polygon

SOURCE = Path(__file__).resolve().parents[1] / "src/spacemap/spatial_distance_acceleration.py"
SPEC = importlib.util.spec_from_file_location("spatial_acceleration_test", SOURCE)
acceleration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceleration)


@pytest.fixture
def torch_module():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_distances_match_geos_with_holes_and_small_blocks(torch_module, device):
    if device == "cuda" and not torch_module.cuda.is_available():
        pytest.skip("CUDA device not available in this process")
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                       holes=[[(30, 30), (70, 30), (70, 70), (30, 70)]]).boundary
    xy = np.random.default_rng(5).uniform(-20, 120, size=(100, 2))
    xy = np.vstack([xy, [[0, 0], [50, 50], [30, 30], [50, 30]]])
    starts, ends = acceleration.boundary_segments(boundary)
    result = acceleration.tensor_segment_distances(torch_module, xy, starts, ends, device, pair_budget=31)
    expected = shapely.distance(boundary, shapely.points(xy))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-10)


def test_separate_rings_are_not_connected_by_artificial_segments(torch_module):
    boundary = MultiLineString([[(0, 0), (1, 0)], [(10, 0), (11, 0)], [(20, 0), (20, 0)]])
    starts, ends = acceleration.boundary_segments(boundary)
    assert len(starts) == 3
    actual = acceleration.tensor_segment_distances(
        torch_module, np.array([[5.0, 0.0], [20.0, 1.0]]), starts, ends, "cpu"
    )
    np.testing.assert_allclose(actual, [4.0, 1.0])


def test_missing_cuda_uses_cpu(torch_module, monkeypatch):
    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: False)
    engine = acceleration.BoundaryDistanceEngine(requested=True)
    assert engine.device is None
    assert "No CUDA GPU" in engine.fallback_reason
    boundaries = np.array([Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).boundary], dtype=object)
    points = shapely.points([[5, 5], [12, 5]])
    np.testing.assert_allclose(engine.distances(boundaries, np.array([0, 0]), points), [5, 2])


def test_runtime_gpu_failure_recomputes_complete_request_on_cpu(monkeypatch):
    engine = acceleration.BoundaryDistanceEngine(minimum_pairs=0)
    engine.device = "test-device"
    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).boundary
    boundaries = np.array([boundary, boundary], dtype=object)
    calls = []

    def failing_kernel(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return np.array([999.0])
        raise RuntimeError("test out of memory")

    monkeypatch.setattr(acceleration, "tensor_segment_distances", failing_kernel)
    result = engine.distances(boundaries, np.array([0, 1]), shapely.points([[5, 5], [12, 5]]))
    np.testing.assert_allclose(result, [5, 2])
    assert engine.device is None and engine.gpu_groups == 0
    assert "out of memory" in engine.fallback_reason


def test_cuda_engine_uses_gpu_and_matches_geos(torch_module):
    if not torch_module.cuda.is_available():
        pytest.skip("CUDA device not available in this process")
    engine = acceleration.BoundaryDistanceEngine(requested=True, minimum_pairs=0)
    boundaries = np.array([Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).boundary], dtype=object)
    points = shapely.points(np.random.default_rng(4).uniform(-5, 15, (100, 2)))
    expected = shapely.distance(boundaries[0], points)
    np.testing.assert_allclose(engine.distances(boundaries, np.zeros(100, dtype=int), points), expected, rtol=1e-12, atol=1e-10)
    assert engine.gpu_groups == 1 and engine.fallback_reason is None
