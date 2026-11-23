"""Cell-marker transformations shared by exploration and sub-phenotyping."""
import numpy as np
import pandas as pd


def transform(frame, markers, mode='auto', cofactor=5.):
    """Arcsinh raw measurements once; corrected columns are already transformed."""
    if mode not in ('auto', 'stored'):
        raise ValueError('Choose automatic arcsinh or stored values.')
    if isinstance(cofactor, bool) or not np.isfinite(cofactor) or cofactor <= 0:
        raise ValueError('Choose a positive arcsinh cofactor.')
    values = frame[markers].apply(pd.to_numeric, errors='coerce').astype('float64')
    values = values.where(np.isfinite(values))
    raw = [m for m in markers if not m.endswith('_corr')] if mode == 'auto' else []
    values[raw] = np.arcsinh(values[raw] / cofactor)
    return values
