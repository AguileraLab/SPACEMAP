"""Infer column types from saved values and edits, without column-name rules."""

import pyarrow as pa


def plain(dtype):
    return dtype.value_type if pa.types.is_dictionary(dtype) else dtype


def array(values, dtype):
    """Convert mixed Python/sidecar values without losing text or missing cells."""
    import pandas as pd

    dtype = plain(dtype)
    try:
        if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
            # Arrow's typed constructor can truncate Python floats to ints;
            # a safe cast must reject that so inference can choose a float.
            return pa.array(values, from_pandas=True).cast(dtype)
        return pa.array(values, type=dtype, from_pandas=True)
    except (pa.ArrowException, ValueError, TypeError):
        pass
    values = [None if pd.isna(v) else v for v in values]
    if pa.types.is_null(dtype):
        return pa.array(values, type=dtype)
    if pa.types.is_boolean(dtype):
        converted = []
        for value in values:
            if value is None:
                converted.append(None)
            elif str(value).lower() in ('true', '1', 'yes'):
                converted.append(True)
            elif str(value).lower() in ('false', '0', 'no'):
                converted.append(False)
            else:
                raise ValueError(f'Not a boolean: {value}')
        return pa.array(converted, type=dtype)
    # Sidecar values are serialized as text, including numeric edits. Let
    # Arrow parse them using the effective type, instead of guessing from the
    # Python type of the first value (which may be an old float or a new label).
    text = pa.array([None if v is None else str(v) for v in values], type=pa.string())
    return text.cast(dtype)


def with_values(dtype, values):
    """Widen a content-derived schema to accommodate current nonempty edits.

    Text edits promote numeric columns to text; numeric edits remain numeric.
    No scan of the entire dataset is needed for each manual assignment.
    """
    dtype = plain(dtype)
    values = list(dict.fromkeys(v for v in values if v is not None and str(v).strip()))
    if not values:
        return dtype
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return dtype
    if pa.types.is_null(dtype):
        candidates = [pa.int64(), pa.uint64(), pa.float64(), pa.bool_()]
    elif pa.types.is_integer(dtype):
        candidates = [dtype, pa.int64(), pa.uint64(), pa.float64()]
    else:
        candidates = [dtype]
    for candidate in candidates:
        try:
            array(values, candidate)
            return candidate
        except (pa.ArrowException, ValueError, OverflowError):
            pass
    return pa.string()
