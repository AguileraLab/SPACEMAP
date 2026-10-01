#!/usr/bin/env python3
"""Exercise the installed scientific runtime using small synthetic inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def check_runtime(device: str, cellpose_weights: Path | None = None) -> dict:
    import numpy as np
    import shapely
    import torch
    import torchvision
    from shapely.geometry import Polygon
    from spacemap.spatial_distance_acceleration import BoundaryDistanceEngine

    torch.set_num_threads(2)
    report = {
        "system": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "spacemap": version("spacemap-ai"),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda, "requested_device": device,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; this is a failed GPU validation, not a skip.")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))

    # Exercise compiled operators, not just imports or a CUDA availability flag.
    x = torch.arange(64, dtype=torch.float64).reshape(8, 8) / 64
    expected = x @ x.T
    actual = x.to(device) @ x.T.to(device)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-12)
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    assert torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() == [0, 2]

    # Check SPACEMAP's real distance engine against independent GEOS results.
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                       holes=[[(30, 30), (70, 30), (70, 70), (30, 70)]]).boundary
    points = shapely.points(np.random.default_rng(5).uniform(-20, 120, (256, 2)))
    engine = BoundaryDistanceEngine(requested=device == "cuda", minimum_pairs=0)
    distances = engine.distances(np.array([boundary], dtype=object), np.zeros(len(points), dtype=int), points)
    np.testing.assert_allclose(distances, shapely.distance(boundary, points), rtol=1e-12, atol=1e-10)
    if device == "cuda":
        assert engine.gpu_groups == 1 and engine.fallback_reason is None, engine.summary()
        torch.cuda.synchronize()
    report["distance_backend"] = engine.summary()["backend"]
    report["compiled_operators"] = "passed"

    if cellpose_weights is not None:
        if not cellpose_weights.is_file():
            raise FileNotFoundError("Pass an existing Cellpose 3 model file to --cellpose-weights.")
        from spacemap.segmentation_cellpose import CellposeSegmenter, SegmentationConfig

        config = SegmentationConfig()
        config.use_gpu = device == "cuda"
        config.cellpose_model_path = str(cellpose_weights.resolve())
        config.channels = [0, 0]
        config.diameter = 20
        config.cellpose_batch_size = 1
        segmenter = CellposeSegmenter(config)
        net_device = next(segmenter.model.net.parameters()).device.type
        assert net_device == device, f"Cellpose used {net_device}, expected {device}"
        yy, xx = np.mgrid[:128, :128]
        synthetic = np.zeros((128, 128), dtype=np.float32)
        for cy, cx in ((32, 32), (32, 96), (96, 32), (96, 96)):
            synthetic += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 72).astype(np.float32)
        masks, flows, styles = segmenter.segment(synthetic[..., None])
        assert masks.shape == synthetic.shape and np.issubdtype(masks.dtype, np.integer)
        assert np.isfinite(styles).all() and all(np.isfinite(f).all() for f in flows)
        if device == "cuda":
            torch.cuda.synchronize()
        report["cellpose"] = {"version": version("cellpose"), "device": net_device,
                              "cells": int(np.count_nonzero(np.unique(masks))),
                              "weights_sha256": hashlib.sha256(cellpose_weights.read_bytes()).hexdigest(), "cudnn_enabled": torch.backends.cudnn.enabled}
    else:
        report["cellpose"] = "not run; supply --cellpose-weights to exercise segmentation"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cellpose-weights", type=Path,
                        help="Existing Cellpose 3 weights; no models are downloaded by this check.")
    parser.add_argument("--report", type=Path, help="Optional JSON result; contains versions, not local paths.")
    args = parser.parse_args()
    report = check_runtime(args.device, args.cellpose_weights)
    result = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(result, encoding="utf-8")
    print(result, end="")
    print("PASS: synthetic runtime validation; this does not assess biological segmentation accuracy.")


if __name__ == "__main__":
    main()
