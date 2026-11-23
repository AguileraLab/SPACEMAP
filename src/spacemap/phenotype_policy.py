"""Explicit class switches, independent of manual labels and fallback names."""
import numpy as np


def allowed(names, excluded=()):
    blocked = {str(name).strip().casefold() for name in excluded}
    return np.asarray([str(name).strip().casefold() not in blocked for name in names], dtype=bool)


def validate_names(value):
    if (not isinstance(value, list) or len(value) > 1000 or
            any(not isinstance(v, str) or not v.strip() or len(v) > 200 or
                any(ord(c) < 32 for c in v) for v in value)):
        raise ValueError('Choose phenotype names from the list.')
    return sorted(set(v.strip() for v in value))


def prediction_positions(classes, excluded=(), mapping=None):
    mapping = mapping or {}
    keep = allowed(classes, excluded) & allowed([mapping.get(str(c), str(c)) for c in classes], excluded)
    positions = np.flatnonzero(keep)
    if not len(positions):
        raise ValueError('All model phenotypes are switched off for prediction. Turn on at least one Predict checkbox.')
    return positions
