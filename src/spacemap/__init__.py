"""
SPACEMAP: A Spatial Multiplex Imaging Analysis Pipeline

A comprehensive, publication-ready pipeline for spatial multiplex imaging analysis,
from image registration to advanced immune phenotyping.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spacemap-ai")
except PackageNotFoundError:
    __version__ = "0+unknown"
__author__ = "Bassel Dawod and Todd A. Aguilera"
__email__ = "basbees@gmail.com"

# Patch skimage.feature.plot_matches for scikit-image 0.25+ compatibility with palom
try:
    import skimage.feature
    if not hasattr(skimage.feature, 'plot_matches'):
        def wrapper_plot_matches(*args, **kwargs):
            names = ['ax', 'image1', 'image2', 'keypoints0', 'keypoints1', 'matches', 'keypoints_color', 'matches_color', 'only_matches', 'alignment']
            bound = {}
            for i, val in enumerate(args):
                if i < len(names):
                    bound[names[i]] = val
            bound.update(kwargs)
            ax = bound.get('ax')
            image1 = bound.get('image1')
            image2 = bound.get('image2')
            keypoints0 = bound.get('keypoints0')
            keypoints1 = bound.get('keypoints1')
            matches = bound.get('matches')
            other_kwargs = {}
            for k in ['keypoints_color', 'matches_color', 'only_matches', 'alignment']:
                if k in bound:
                    other_kwargs[k] = bound[k]
            return skimage.feature.plot_matched_features(
                image1, image2, keypoints0=keypoints0, keypoints1=keypoints1,
                matches=matches, ax=ax, **other_kwargs
            )
        skimage.feature.plot_matches = wrapper_plot_matches
except Exception:
    pass

# Patch skimage.transform geometric transformations to fix 2D translation compatibility with scikit-image 0.25+
try:
    import skimage.transform
    import numpy as np
    import inspect
    def patch_transform_class(cls):
        orig_init = cls.__init__
        def patched_init(self, *args, **kwargs):
            sig = inspect.signature(orig_init)
            bound = sig.bind_partial(self, *args, **kwargs)
            if 'translation' in bound.arguments and bound.arguments['translation'] is not None:
                t = np.asarray(bound.arguments['translation'])
                if t.ndim == 2 and t.shape[0] == 1:
                    bound.arguments['translation'] = t[0]
                elif t.ndim > 1:
                    bound.arguments['translation'] = np.squeeze(t)
            return orig_init(*bound.args, **bound.kwargs)
        cls.__init__ = patched_init

    for cls in [skimage.transform.AffineTransform, skimage.transform.EuclideanTransform, skimage.transform.SimilarityTransform]:
        patch_transform_class(cls)
except Exception:
    pass

# Submodules load on first use (PEP 562). Importing them all here made
# ``import spacemap`` pull in PyTorch, Cellpose and SCIMAP -- about a minute on a
# cold start -- for every worker process, test and tool, whether it needed them
# or not. ``spacemap.segmentation_cellpose`` and ``from spacemap import
# ph_leiden_utils`` work exactly as before.
import importlib as _importlib

_SUBMODULES = (
    "config", "utils", "registration", "segmentation_cellpose", "segmentation_deepcell",
    "ph_leiden_utils", "ph_som_utils", "ph_thres_utils", "ph_scimap_utils", "ml_mr_utils",
    "ml_consensus", "annotation_utils", "batch_correction_utils", "comparison_utils", "acc_utils",
)
# These were set to None when their dependencies were missing; keep that contract.
_NONE_WHEN_UNAVAILABLE = ("registration", "annotation_utils", "batch_correction_utils",
                          "comparison_utils", "acc_utils")


def __getattr__(name):
    if name in _SUBMODULES:
        try:
            module = _importlib.import_module(f".{name}", __name__)
        except (ImportError, TypeError, RuntimeError):
            if name in _NONE_WHEN_UNAVAILABLE:
                globals()[name] = None
                return None
            raise
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))


# Version and metadata
__all__ = [
    '__version__',
    'config',
    'utils',
    'registration',
    'segmentation_cellpose',
    'segmentation_deepcell',
    'ph_leiden_utils',
    'ph_som_utils',
    'ph_thres_utils',
    'ph_scimap_utils',
    'ml_mr_utils',
    'ml_consensus',
    'annotation_utils',
    'batch_correction_utils',
    'comparison_utils',
    'acc_utils'
]
