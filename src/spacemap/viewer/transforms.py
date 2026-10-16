"""Invertible histogram views. Raw feature values remain the stored thresholds.

Unlike tie-score normalization, these transforms never clip values. Parameters
are fitted once to the selected sample; a chain that loses numerical precision
is rejected instead of silently changing the raw threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

OPTIONS = ("Raw", "Log10", "Arcsinh", "Z1", "CDF", "NLog")


@dataclass
class Transform:
    stages: list[dict]

    @classmethod
    def fit(cls, values, names=(), cofactor=5.0):
        names = list(names)
        if len(names) > 8 or any(name not in OPTIONS for name in names):
            raise ValueError("Choose up to eight steps from Raw, Log10, Arcsinh, Z1, CDF and NLog.")
        if not np.isfinite(cofactor) or cofactor <= 0:
            raise ValueError("Arcsinh cofactor must be positive and finite.")
        raw = np.asarray(values, dtype=np.float64)
        raw = raw[np.isfinite(raw)]
        if not raw.size:
            raise ValueError("No finite values in this sample.")
        data = raw.copy()
        result = cls([])
        for name in names:
            if name == "Raw":
                continue
            stage = {"name": name}
            if name == "Log10":
                stage["offset"] = max(1.0, 1.0 - float(data.min()))
            elif name == "Arcsinh":
                stage["cofactor"] = float(cofactor)
            elif name == "Z1":
                stage.update(mean=float(data.mean()), scale=float(data.std()) or 1.0)
            data = cls([stage]).forward(data)
            result.stages.append(stage)
        restored = result.inverse(data)
        if not np.allclose(restored, raw, atol=1e-6, rtol=1e-10):
            raise ValueError("This combination loses precision. Remove CDF, or add Arcsinh before Z1 and CDF.")
        return result

    def _apply(self, values, inverse):
        data = np.asarray(values, dtype=np.float64)
        with np.errstate(all="ignore"):
            for stage in reversed(self.stages) if inverse else self.stages:
                name = stage["name"]
                if name == "Log10":
                    data = np.power(10.0, data) - stage["offset"] if inverse else np.log10(data + stage["offset"])
                elif name == "Arcsinh":
                    data = np.sinh(data) * stage["cofactor"] if inverse else np.arcsinh(data / stage["cofactor"])
                elif name == "Z1":
                    data = data * stage["scale"] + stage["mean"] if inverse else (data - stage["mean"]) / stage["scale"]
                elif name == "CDF":
                    data = ndtri(data) if inverse else ndtr(data)
                    if not inverse and np.any((data <= 0) | (data >= 1)):
                        raise ValueError(
                            "CDF cannot keep these values distinct. Add Arcsinh and Z1 first, or remove CDF."
                        )
                elif name == "NLog":
                    data = np.power(10.0, -data) if inverse else -np.log10(data)
        if not np.all(np.isfinite(data)):
            raise ValueError(
                "This combination cannot convert the value back. Try removing the last step. NLog needs values above zero."
            )
        return data

    def forward(self, values):
        return self._apply(values, False)

    def inverse(self, values):
        return self._apply(values, True)


def histogram(values, transform, threshold, bins=128):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    scaled = transform.forward(values)
    counts, edges = np.histogram(scaled, bins=bins)
    return {
        "counts": counts.tolist(),
        "axis": [float(edges[0]), float(edges[-1])],
        "transformed_threshold": float(transform.forward(threshold)),
        "threshold": float(threshold),
        "total": int(values.size),
        "positive": int((values > threshold).sum()),
        "stages": transform.stages,
    }


def automatic_threshold(values, transform, method, params=None):
    from spacemap.thresholding_engine import calculate
    found = calculate(values, method, params=params)
    raw = found["threshold"]
    return {"raw": raw, "value": float(transform.forward(raw)), "fallback": "none"}
