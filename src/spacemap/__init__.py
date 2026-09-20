"""
SPACEMAP: A Spatial Multiplex Imaging Analysis Pipeline

A comprehensive, publication-ready pipeline for spatial multiplex imaging analysis,
from image registration to advanced immune phenotyping.
"""

__version__ = "1.0.0"
__author__ = "SPACEMAP Development Team"
__email__ = "contact@spacemap.org"

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

# Core modules
from . import config
from . import utils
try:
    from . import registration
except (ImportError, TypeError, RuntimeError):
    registration = None


# Segmentation modules
try:
    from . import segmentation_cellpose
except (ImportError, TypeError, RuntimeError):
    pass
try:
    from . import segmentation_deepcell
except (ImportError, TypeError, RuntimeError):
    pass

# Phenotyping modules
try:
    from . import ph_leiden_utils
except (ImportError, TypeError, RuntimeError):
    pass
try:
    from . import ph_som_utils
except (ImportError, TypeError, RuntimeError):
    pass
try:
    from . import ph_thres_utils
except (ImportError, TypeError, RuntimeError):
    pass
try:
    from . import ph_scimap_utils
except (ImportError, TypeError, RuntimeError):
    pass

# Machine learning modules
try:
    from . import ml_mr_utils
except (ImportError, TypeError, RuntimeError):
    pass
try:
    from . import ml_consensus
except (ImportError, TypeError, RuntimeError):
    pass

# Utility modules
try:
    from . import annotation_utils
except (ImportError, TypeError, RuntimeError):
    annotation_utils = None
try:
    from . import batch_correction_utils
except (ImportError, TypeError, RuntimeError):
    batch_correction_utils = None
try:
    from . import comparison_utils
except (ImportError, TypeError, RuntimeError):
    comparison_utils = None
try:
    from . import acc_utils
except (ImportError, TypeError, RuntimeError):
    acc_utils = None


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
