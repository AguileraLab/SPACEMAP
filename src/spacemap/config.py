"""
Configuration for the SPACEMAP-V4 pipeline.
"""

import logging
from pathlib import Path





class ProjectConfig:
    """
    A central configuration class for the project.
    """
    def __init__(self, project_root=None, create_directories=False):
        if project_root:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = None
        
        if self.project_root:
            # Core data paths
            self.data_dir = self.project_root / 'data'
            self.codex_dir = self.data_dir / 'codex'
            self.he_dir = self.data_dir / 'he'
            self.boundaries_dir = self.data_dir / 'boundaries'
            self.seg_model_dir = self.data_dir / 'seg_model'
            
            # Annotation paths (as inputs)
            self.annotations_dir = self.data_dir / 'annotations'
            self.raw_tissue_dir = self.annotations_dir / 'stroma_epi_bkg'
            self.removal_dir = self.annotations_dir / 'removal_annotations'
            self.endothelia_dir = self.annotations_dir / 'endothelia'

            # Output paths
            self.output_dir = self.project_root / 'outputs'
            self.registered_dir = self.output_dir / 'registered'
            self.background_subtracted_dir = self.output_dir / 'background_subtracted'
            self.rois_dir = self.output_dir / 'rois'
            self.segmentation_dir = self.output_dir / 'segmentation'
            
            # Processed annotation output paths
            self.processed_annotations_dir = self.output_dir / 'processed_annotations'
            self.stage1a_merged_tissue_dir = self.processed_annotations_dir / 'stage1a_merged_tissue'
            self.stage1b_compartments_zones_dir = self.processed_annotations_dir / 'stage1b_compartments_zones'
            
            # Final annotated CSV output paths
            self.annotated_csv_dir = self.output_dir / 'annotated_csv'
            self.batch_corrected_dir = self.output_dir / 'batch_corrected'
            self.phenotyping_dir = self.output_dir / 'phenotyping'
            self.spatial_analysis_dir = self.output_dir / 'spatial_analysis'

            if create_directories:
                self.create_dirs()

    def create_dirs(self):
        """Create all necessary directories for active project."""
        if not self.project_root:
            return
        dirs = [
            self.data_dir, self.codex_dir, self.he_dir, self.boundaries_dir,
            self.seg_model_dir, self.annotations_dir, self.raw_tissue_dir,
            self.removal_dir, self.endothelia_dir, self.output_dir,
            self.registered_dir, self.background_subtracted_dir, self.rois_dir,
            self.segmentation_dir, self.processed_annotations_dir,
            self.stage1a_merged_tissue_dir, self.stage1b_compartments_zones_dir,
            self.annotated_csv_dir, self.batch_corrected_dir,
            self.phenotyping_dir, self.spatial_analysis_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"ProjectConfig(project_root='{self.project_root}')"


# Default configuration reference (lazy initialized when needed)
default_config = None


# Convenience functions for common operations
def get_config(base_dir=None, create_directories=False):
    """Get a configuration instance."""
    return ProjectConfig(base_dir, create_directories=create_directories)


def setup_logging(name, level='INFO'):
    """Setup logging for pipeline components."""
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def log_step(step_name, logger=None):
    """Context manager for logging processing steps with timing."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _log_step():
        start_time = time.time()
        if logger:
            logger.info(f"Starting: {step_name}")
        else:
            print(f"Starting: {step_name}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if logger:
                logger.info(f"Completed: {step_name} (took {elapsed:.1f}s)")
            else:
                print(f"Completed: {step_name} (took {elapsed:.1f}s)")
    
    return _log_step()


# Export main components
__all__ = [
    'ProjectConfig',
    'default_config', 
    'get_config',
    'setup_logging',
    'log_step'
]
