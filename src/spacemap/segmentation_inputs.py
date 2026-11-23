"""Shared nuclear/border projections for previews, training and segmentation."""

import numpy as np
import re


def dapi_anchor(names):
    """The selected DAPI channel, preserving its exact image name."""
    return next((name for name in names if name.casefold() == 'dapi'),
                next((name for name in names if re.match(r'^dapi(?:\b|[_ -]|\d)', name, re.I)), None))


def marker_subsets(nuclear, borders, seed):
    """Reproducible missing-input scenarios; DAPI stays in every nuclear input.

    Drop H&E RGB as a group. Partial subsets vary across tiles without depending
    on channel order, and never introduce a signal absent at inference time.
    """
    anchor = dapi_anchor(nuclear)
    if not anchor:
        raise ValueError('Train for missing markers requires DAPI selected as a Nucleus marker.')
    he = lambda name: bool(re.match(r'^(he(?:\b|[_ -])|h&e|hematoxylin|haematoxylin|eosin)', name, re.I))
    without_he = ([n for n in nuclear if not he(n)], [n for n in borders if not he(n)])
    rng = np.random.default_rng(seed)
    optional = sorted(set(nuclear + borders) - {anchor}, key=str.casefold)
    # H&E is removed entirely for the partial-panel recipe too.
    kept = {anchor, *(name for name in optional if not he(name) and rng.random() < 0.5)}
    return {
        'full': (nuclear, borders),
        'without_he': without_he,
        'fewer_markers': ([n for n in nuclear if n in kept], [n for n in borders if n in kept]),
        'dapi_only': ([anchor], []),
    }


def maximum_projection(planes):
    """Pixel-wise maximum, without allocating a stack of all selected markers."""
    planes = iter(planes)
    first = next(planes, None)
    if first is None:
        raise ValueError('Choose at least one nuclear marker.')
    result = np.array(first, dtype=np.float32, copy=True)
    if result.ndim != 2:
        raise ValueError('Segmentation markers must be two-dimensional planes.')
    for plane in planes:
        if np.shape(plane) != result.shape:
            raise ValueError('Segmentation markers must have matching image dimensions.')
        np.maximum(result, plane, out=result)
    return result


def cellpose_input(nuclear_planes, border_planes=()):
    """H×W×3: maximum border signal, empty, maximum nuclear signal.

    Without border markers, the nuclear projection supplies both model inputs.
    Original measurement channels are never modified or independently rescaled.
    """
    nucleus = maximum_projection(nuclear_planes)
    borders = iter(border_planes)
    first = next(borders, None)
    if first is None:
        border = nucleus
    else:
        from itertools import chain
        border = maximum_projection(chain([first], borders))
        if border.shape != nucleus.shape:
            raise ValueError('Nuclear and border markers must have matching image dimensions.')
    return np.stack((border, np.zeros_like(nucleus), nucleus), axis=-1)


def resolve_nuclear_indices(names, selected):
    """Accept legacy single-marker calls and explicit multi-marker selections."""
    single = isinstance(selected, str)
    chosen = [selected] if single else selected
    if not isinstance(chosen, (list, tuple)) or not chosen or not all(isinstance(n, str) and n for n in chosen):
        raise ValueError('Choose at least one nuclear marker.')
    lookup = {name.casefold(): i for i, name in enumerate(names) if isinstance(name, str)}
    missing = [name for name in chosen if name.casefold() not in lookup]
    if missing:
        # Preserve the legacy default's support for names such as DAPI_1.
        fallback = next((i for i, name in enumerate(names) if 'dapi' in str(name).lower()), None)
        if single and fallback is not None:
            return [fallback]
        raise ValueError('Nuclear markers not found in image: ' + ', '.join(missing))
    return list(dict.fromkeys(lookup[name.casefold()] for name in chosen))
