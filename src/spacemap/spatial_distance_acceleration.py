"""Optional CUDA acceleration for distances to polygon boundary segments.

GEOS still determines polygon ownership, inside/outside, and path containment.
Only the numeric point-to-segment reduction runs on CUDA, using float64 and
bounded tensor blocks. PyTorch is imported only when GPU use is requested.
"""

from __future__ import annotations

import numpy as np
import shapely


def boundary_segments(boundary):
    """Return segment endpoints, without connecting separate polygon rings."""
    coordinates, rings = shapely.get_coordinates(shapely.get_parts(boundary), return_index=True)
    connected = rings[1:] == rings[:-1]
    return coordinates[:-1][connected], coordinates[1:][connected]


def tensor_segment_distances(torch, points, starts, ends, device, pair_budget=2_000_000):
    """Exact Euclidean point-to-segment formula, reduced in bounded blocks."""
    if not len(starts):
        return np.full(len(points), np.nan)
    result = np.empty(len(points), dtype=float)
    with torch.no_grad():
        starts_device = torch.as_tensor(starts, dtype=torch.float64, device=device)
        vectors = torch.as_tensor(ends, dtype=torch.float64, device=device) - starts_device
        lengths_squared = (vectors * vectors).sum(dim=1)
        # A zero-length segment is a point: projection zero yields its distance.
        denominators = torch.where(lengths_squared > 0, lengths_squared, torch.ones_like(lengths_squared))
        point_chunk = min(1024, max(int(pair_budget), 1))
        segment_chunk = max(int(pair_budget) // point_chunk, 1)
        for start in range(0, len(points), point_chunk):
            xy = torch.as_tensor(points[start:start + point_chunk], dtype=torch.float64, device=device)
            best = torch.full((len(xy),), float("inf"), dtype=torch.float64, device=device)
            for first in range(0, len(starts), segment_chunk):
                last = first + segment_chunk
                offset = xy[:, None, :] - starts_device[None, first:last, :]
                direction = vectors[first:last]
                projection = ((offset * direction[None, :, :]).sum(dim=2)
                              / denominators[None, first:last]).clamp_(0.0, 1.0)
                residual = offset - projection[:, :, None] * direction[None, :, :]
                squared = (residual * residual).sum(dim=2)
                best = torch.minimum(best, squared.min(dim=1).values)
            result[start:start + len(xy)] = best.sqrt().cpu().numpy()
    return result


class BoundaryDistanceEngine:
    """Use CUDA for large reductions and GEOS for small groups or failures."""

    def __init__(self, requested=False, minimum_pairs=250_000):
        self.requested = bool(requested)
        self.minimum_pairs = int(minimum_pairs)
        self.torch = None
        self.device = None
        self.device_name = None
        self.fallback_reason = None
        self.gpu_groups = 0
        if not requested:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                self.fallback_reason = "No CUDA GPU is visible to this app process."
                return
            self.torch = torch
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.device_name = torch.cuda.get_device_name(self.device)
        except Exception as error:
            self.fallback_reason = f"CUDA initialization failed: {error}"

    def distances(self, boundaries, owners, points):
        if self.device is None or len(points) == 0:
            return shapely.distance(boundaries[owners], points)
        result = np.empty(len(points), dtype=float)
        previous_gpu_groups = self.gpu_groups
        try:
            # Sort once rather than scan all points separately for every region.
            order = np.argsort(owners, kind="stable")
            cuts = np.r_[0, np.flatnonzero(np.diff(owners[order])) + 1, len(order)]
            small_groups = []
            for first, last in zip(cuts[:-1], cuts[1:]):
                positions = order[first:last]
                boundary = boundaries[int(owners[positions[0]])]
                if len(positions) * shapely.get_num_coordinates(boundary) < self.minimum_pairs:
                    small_groups.append(positions)
                    continue
                starts, ends = boundary_segments(boundary)
                coordinates = shapely.get_coordinates(points[positions])
                result[positions] = tensor_segment_distances(
                    self.torch, coordinates, starts, ends, self.device
                )
                self.gpu_groups += 1
            if small_groups:
                positions = np.concatenate(small_groups)
                result[positions] = shapely.distance(boundaries[owners[positions]], points[positions])
            return result
        except Exception as error:
            # Recompute the entire request with GEOS; partial GPU results must
            # never leave missing or inconsistent values in the output.
            self.fallback_reason = f"GPU distance calculation failed; used CPU: {error}"
            self.gpu_groups = previous_gpu_groups
            self.device = None
            return shapely.distance(boundaries[owners], points)

    def summary(self):
        return {
            "requested": self.requested,
            "device": self.device_name,
            "gpu_groups": self.gpu_groups,
            "fallback_reason": self.fallback_reason,
            "backend": "GPU + CPU" if self.gpu_groups else "CPU",
        }
