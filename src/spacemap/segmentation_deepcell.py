"""
DeepCell segmentation utilities for SPACEMAP pipeline.

This module provides functionality for cell and nuclear segmentation using DeepCell Mesmer.
It follows the same structure as the Cellpose segmentation module for consistency.
"""

import os
import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET
import cv2
import json
import pandas as pd
from datetime import datetime
from skimage.measure import regionprops_table
import rasterio.features
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

# Import DeepCell
try:
    from deepcell.applications import Mesmer
    DEEPCELL_AVAILABLE = True
    
    # Set up authentication token if needed
    if "DEEPCELL_ACCESS_TOKEN" not in os.environ:
        # You may need to set this token for DeepCell API access
        # os.environ["DEEPCELL_ACCESS_TOKEN"] = "your_token_here"
        pass
        
except (ImportError, TypeError, Exception) as e:
    DEEPCELL_AVAILABLE = False


class DeepCellConfig:
    """Configuration class for DeepCell segmentation parameters."""
    
    def __init__(self):
        # DeepCell Mesmer specific parameters
        self.compartment_mode = "both"  # Options: "nuclear", "whole-cell", "both"
        self.primary_output_mask = "cell"  # Options: "cell", "nuclear" (for "both" mode)
        self.batch_size = 4
        self.image_mpp = None  # Microns per pixel (optional)
        
        # Membrane marker configuration
        self.merge_membrane_markers = True  # Use max projection of all available markers
        
        # General segmentation parameters
        self.tile_size = (256, 256)
        self.n_pyramid_levels = 3
        self.save_geojson = True
        self.save_npy = False
        self.save_png = False
        
        # Optional masking with input GeoJSON
        self.use_input_geojson = False
        self.input_geojson_path = "./data/json/input_mask.geojson"


def get_channel_names(ome_tiff_path):
    """Extract channel names from OME-TIFF metadata or fallback to defaults."""
    try:
        with tiff.TiffFile(ome_tiff_path) as tif:
            ome_xml = tif.ome_metadata
            root = ET.fromstring(ome_xml)
            namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            channels = root.findall('.//ome:Channel', namespaces)
            channel_names = [channel.get('Name') for channel in channels]
            if not channel_names or None in channel_names:
                raise ValueError("Channel names missing.")
            return channel_names
    except Exception as e:
        print(f"Error extracting channel names: {e}")
        print("Using default channel names.")
        with tiff.TiffFile(ome_tiff_path) as tif:
            return [f'Channel_{i}' for i in range(len(tif.pages))]


def extract_features(mask, images, channel_names):
    """
    Extract features for each segmented cell or nucleus.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Segmentation mask with labeled regions
    images : list of numpy.ndarray
        List of channel images
    channel_names : list of str
        Names of the channels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted features for each cell/nucleus
    """
    print(f"Segmentation mask shape: {mask.shape}")
    print(f"Processing {len(images)} channels")

    # Ensure mask is integer type
    mask_int = mask.astype(np.int32)
    if np.max(mask_int) == 0:
        print("Warning: Empty mask, no objects to extract features from.")
        return pd.DataFrame()

    print("Extracting morphological features...")
    # First, get all morphological properties. This is done once.
    morph_props = regionprops_table(
        mask_int,
        properties=[
            'label', 'area', 'major_axis_length', 'minor_axis_length',
            'eccentricity', 'solidity', 'extent', 'orientation', 'centroid'
        ]
    )
    morph_df = pd.DataFrame(morph_props)
    morph_df.rename(columns={'centroid-0': 'y_centroid', 'centroid-1': 'x_centroid'}, inplace=True)

    # Create aligned images with matching dimensions to the mask
    aligned_images = []
    aligned_channels = []
    for image, channel_name in zip(images, channel_names):
        try:
            image = image.astype(np.float32)
            if image.shape != mask.shape:
                resized_img = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                aligned_images.append(resized_img)
                aligned_channels.append(channel_name)
            else:
                aligned_images.append(image)
                aligned_channels.append(channel_name)
        except Exception as e:
            print(f"Skipping channel {channel_name} due to error: {e}")

    # Now, extract intensity features for each channel
    intensity_features = []
    for image, channel_name in zip(aligned_images, aligned_channels):
        print(f"Extracting intensity from channel {channel_name}...")
        intensity_props = regionprops_table(
            mask_int,
            intensity_image=image,
            properties=['label', 'mean_intensity']
        )
        intensity_df = pd.DataFrame(intensity_props)
        intensity_df.rename(columns={'mean_intensity': channel_name}, inplace=True)
        intensity_features.append(intensity_df.set_index('label'))

    # Combine morphological and intensity features
    combined_df = morph_df.set_index('label')
    if intensity_features:
        combined_df = combined_df.join(intensity_features)
    
    return combined_df.reset_index()


def generate_geojson_from_mask(mask, output_path, mask_type="cell"):
    """
    Generate GeoJSON file from segmentation mask.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Segmentation mask with labeled regions
    output_path : str
        Path to save the GeoJSON file
    mask_type : str
        Type of mask ("cell" or "nuclear") for naming
        
    Returns:
    --------
    str
        Path to the saved GeoJSON file
    """
    # Same reasoning as the Cellpose exporter: no "skip if the file exists"
    # guard, because reaching this point means fresh masks were just produced
    # and any existing file is a stale export that would disagree with the
    # features CSV written alongside it.
    print(f"Exporting {mask_type} boundaries as GeoJSON...")
    
    try:
        # Ensure mask is integer type
        mask_int = mask.astype(np.int32)
        
        # Check if mask is empty
        if np.max(mask_int) == 0:
            print(f"Warning: Empty mask, no {mask_type} boundaries to export")
            # Create an empty GeoJSON
            features = []
            geojson_dict = {"type": "FeatureCollection", "features": features}
            with open(output_path, 'w') as f:
                json.dump(geojson_dict, f, indent=2)
            print(f"Empty {mask_type} boundaries GeoJSON saved to: {output_path}")
            return output_path
        
        # Extract shapes from mask
        results = rasterio.features.shapes(mask_int, mask=(mask_int > 0), connectivity=8)
        features = []
        
        # Process each shape
        for geom, value in results:
            if value == 0:
                continue
            try:
                # Round coordinates and ensure they are standard integers
                geom['coordinates'] = [[[int(round(coord)) for coord in point] for point in ring] for ring in geom['coordinates']]
                
                poly = shape(geom)
                
                # Skip invalid geometries
                if not poly.is_valid:
                    print(f"Warning: Invalid polygon for {mask_type} {value}, attempting to fix...")
                    poly = poly.buffer(0)  # Buffer with 0 distance often fixes invalid polygons
                    if not poly.is_valid:
                        print(f"Warning: Could not fix invalid polygon for {mask_type} {value}, skipping")
                        continue
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "label": int(value),
                        "area": float(poly.area),
                    },
                    "geometry": geom # Use the modified geom dictionary directly
                }
                features.append(feature)
            except Exception as e:
                print(f"Error processing polygon for {mask_type} {value}: {e}")

        # Create GeoJSON
        geojson_dict = {"type": "FeatureCollection", "features": features}
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as a single line with no extra spaces
        with open(output_path, 'w') as f:
            json.dump(geojson_dict, f, separators=(',', ':'))
                    
        print(f"{mask_type.capitalize()} boundaries GeoJSON saved to: {output_path}")
        print(f"Exported {len(features)} {mask_type} boundaries")
        return output_path
    
    except Exception as e:
        print(f"Error generating GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_tissue_mask(dapi_image, geojson_path):
    """
    Create a tissue mask from GeoJSON annotations.
    
    Parameters:
    -----------
    dapi_image : numpy.ndarray
        DAPI image to get dimensions
    geojson_path : str
        Path to GeoJSON file with tissue annotations
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where 1 indicates tissue regions
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    tissue_polygons = []
    exclude_polygons = []
    
    for feature in geojson_data['features']:
        annotation_name = feature['properties']['classification']['name'].lower()
        geom = shape(feature['geometry'])
        
        if annotation_name == 'tissue':
            tissue_polygons.append(geom)
        elif annotation_name in ['artefact', 'blood_clot', 'exclude']:
            exclude_polygons.append(geom)
    
    # Create union of exclusion regions
    exclude_union = unary_union(exclude_polygons) if exclude_polygons else None
    
    # Process tissue polygons by removing excluded regions
    processed_polygons = []
    for tissue_poly in tissue_polygons:
        if not tissue_poly.is_empty:
            if exclude_union:
                processed_poly = tissue_poly.difference(exclude_union)
            else:
                processed_poly = tissue_poly
            processed_polygons.append(processed_poly)
    
    # Create mask image
    mask_image = np.zeros(dapi_image.shape, dtype=np.uint8)
    
    for poly in processed_polygons:
        if isinstance(poly, (Polygon, MultiPolygon)):
            polys_to_fill = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            
            for subpoly in polys_to_fill:
                exterior_coords = np.array(subpoly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask_image, [exterior_coords], 1)
                
                # Handle holes
                for interior in subpoly.interiors:
                    interior_coords = np.array(interior.coords, dtype=np.int32)
                    cv2.fillPoly(mask_image, [interior_coords], 0)
    
    return mask_image


class DeepCellSegmenter:
    """DeepCell Mesmer-based cell and nuclear segmentation."""
    
    def __init__(self, config):
        """
        Initialize DeepCell segmenter.
        
        Parameters:
        -----------
        config : DeepCellConfig
            Configuration object with segmentation parameters
        """
        if not DEEPCELL_AVAILABLE:
            raise ImportError("DeepCell is required but not available")
            
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the DeepCell Mesmer model."""
        print("Loading DeepCell Mesmer model...")
        try:
            self.model = Mesmer()
            print("DeepCell Mesmer model loaded successfully")
        except Exception as e:
            print(f"Error loading DeepCell model: {e}")
            raise
    
    def segment(self, image_stack):
        """
        Perform segmentation on image stack.
        
        Parameters:
        -----------
        image_stack : numpy.ndarray
            Image stack for segmentation (H, W, C) - expects 2 channels: nuclear, membrane
            
        Returns:
        --------
        dict
            Dictionary containing segmentation results
        """
        print("Running DeepCell Mesmer segmentation...")
        
        try:
            # Prepare input for DeepCell (expects batch dimension)
            if len(image_stack.shape) == 3:
                image_stack = np.expand_dims(image_stack, 0)
            
            # Ensure float32 and proper normalization
            image_stack = image_stack.astype(np.float32)
            
            # Check if normalization is needed
            if np.max(image_stack) > 1.0 or np.min(image_stack) < 0.0:
                print("Normalizing input to [0,1] range")
                for c in range(image_stack.shape[-1]):
                    channel = image_stack[0, :, :, c]
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:
                        image_stack[0, :, :, c] = (channel - min_val) / (max_val - min_val)
                    else:
                        image_stack[0, :, :, c] = 0
            
            # Run segmentation
            predictions = self.model.predict(
                image_stack,
                batch_size=self.config.batch_size,
                image_mpp=self.config.image_mpp,
                compartment=self.config.compartment_mode
            )
            
            # Extract masks based on compartment mode
            results = {}
            
            if self.config.compartment_mode == "both":
                # Both cell and nuclear segmentation
                cell_mask = predictions[0, :, :, 0]
                nuclear_mask = predictions[0, :, :, 1]
                results['cell'] = cell_mask
                results['nuclear'] = nuclear_mask
                
                # Determine primary mask
                if self.config.primary_output_mask == "nuclear":
                    results['primary'] = nuclear_mask
                    results['primary_type'] = "nuclear"
                else:
                    results['primary'] = cell_mask
                    results['primary_type'] = "cell"
                    
                print(f"Segmentation completed. Found {len(np.unique(cell_mask)) - 1} cells and {len(np.unique(nuclear_mask)) - 1} nuclei.")
                
            elif self.config.compartment_mode == "nuclear":
                # Nuclear segmentation only
                nuclear_mask = predictions[0, :, :, 0]
                results['nuclear'] = nuclear_mask
                results['primary'] = nuclear_mask
                results['primary_type'] = "nuclear"
                print(f"Nuclear segmentation completed. Found {len(np.unique(nuclear_mask)) - 1} nuclei.")
                
            else:  # "whole-cell"
                # Cell segmentation only
                cell_mask = predictions[0, :, :, 0]
                results['cell'] = cell_mask
                results['primary'] = cell_mask
                results['primary_type'] = "cell"
                print(f"Cell segmentation completed. Found {len(np.unique(cell_mask)) - 1} cells.")
            
            return results
            
        except Exception as e:
            print(f"Error during DeepCell segmentation: {e}")
            raise


def check_segmentation_outputs_exist(input_image_path, output_dir, config):
    """
    Check if segmentation outputs already exist for an image.
    
    Parameters:
    -----------
    input_image_path : str
        Path to input image
    output_dir : str
        Output directory
    config : DeepCellConfig
        Configuration object
        
    Returns:
    --------
    dict
        Dictionary with existence status of each output type
    """
    input_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    outputs = {
        'features_csv': False,
        'geojson': False,
        'mask_npy': False,
        'segmentation_png': False,
        'all_exist': False
    }
    
    # Check for features CSV based on compartment mode
    if config.compartment_mode == "both":
        # Check for both cell and nuclear CSV files
        cell_csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
        nuclear_csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
        outputs['features_csv'] = os.path.exists(cell_csv_path) and os.path.exists(nuclear_csv_path)
    elif config.compartment_mode == "nuclear":
        nuclear_csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
        outputs['features_csv'] = os.path.exists(nuclear_csv_path)
    else:  # "whole-cell"
        cell_csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
        outputs['features_csv'] = os.path.exists(cell_csv_path)
    
    # Check for GeoJSON files
    if config.save_geojson:
        if config.compartment_mode == "both":
            cell_geojson_path = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
            nuclear_geojson_path = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
            outputs['geojson'] = os.path.exists(cell_geojson_path) and os.path.exists(nuclear_geojson_path)
        elif config.compartment_mode == "nuclear":
            nuclear_geojson_path = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
            outputs['geojson'] = os.path.exists(nuclear_geojson_path)
        else:
            cell_geojson_path = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
            outputs['geojson'] = os.path.exists(cell_geojson_path)
    else:
        outputs['geojson'] = True  # Not required
    
    # Check other optional outputs
    if config.save_npy:
        if config.compartment_mode == "both":
            cell_npy_path = os.path.join(output_dir, f"{input_image_name}_cell_mask.npy")
            nuclear_npy_path = os.path.join(output_dir, f"{input_image_name}_nuclear_mask.npy")
            outputs['mask_npy'] = os.path.exists(cell_npy_path) and os.path.exists(nuclear_npy_path)
        elif config.compartment_mode == "nuclear":
            nuclear_npy_path = os.path.join(output_dir, f"{input_image_name}_nuclear_mask.npy")
            outputs['mask_npy'] = os.path.exists(nuclear_npy_path)
        else:
            cell_npy_path = os.path.join(output_dir, f"{input_image_name}_cell_mask.npy")
            outputs['mask_npy'] = os.path.exists(cell_npy_path)
    else:
        outputs['mask_npy'] = True  # Not required
        
    if config.save_png:
        png_path = os.path.join(output_dir, f"{input_image_name}_segmentation.png")
        outputs['segmentation_png'] = os.path.exists(png_path)
    else:
        outputs['segmentation_png'] = True  # Not required
    
    # Consider outputs complete if all enabled outputs exist
    outputs['all_exist'] = all([
        outputs['features_csv'],
        outputs['geojson'],
        outputs['mask_npy'],
        outputs['segmentation_png']
    ])
    
    return outputs


def segment_image_deepcell(ome_tiff_path, output_dir, config, markers=None, dapi_channel='DAPI', 
                          overwrite=False):
    """
    Complete segmentation pipeline for a single OME-TIFF image using DeepCell Mesmer.
    
    Parameters:
    -----------
    ome_tiff_path : str
        Path to input OME-TIFF file
    output_dir : str
        Directory to save outputs
    config : DeepCellConfig
        Configuration object
    markers : list of str, optional
        List of marker channel names to use for membrane signal
    dapi_channel : str
        Name of DAPI channel
    overwrite : bool
        Whether to overwrite existing outputs (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing paths to output files and segmentation results
    """
    start_time = datetime.now()
    input_image_name = os.path.splitext(os.path.basename(ome_tiff_path))[0]
    
    # Check if outputs already exist
    if not overwrite:
        existing_outputs = check_segmentation_outputs_exist(ome_tiff_path, output_dir, config)
        if existing_outputs['all_exist']:
            print(f"Outputs already exist for {input_image_name}. Use overwrite=True to reprocess.")
            
            # Load existing results for consistency
            output_files = {}
            
            # Load features CSV(s)
            if config.compartment_mode == "both":
                cell_csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
                nuclear_csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
                if os.path.exists(cell_csv_path):
                    output_files['cell_features_csv'] = cell_csv_path
                if os.path.exists(nuclear_csv_path):
                    output_files['nuclear_features_csv'] = nuclear_csv_path
                # Load primary features based on config
                primary_csv = cell_csv_path if config.primary_output_mask == "cell" else nuclear_csv_path
                if os.path.exists(primary_csv):
                    try:
                        output_files['features_df'] = pd.read_csv(primary_csv)
                        output_files['features_csv'] = primary_csv
                    except Exception as e:
                        print(f"Warning: Could not load existing CSV: {e}")
            else:
                # Single mode
                if config.compartment_mode == "nuclear":
                    csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
                else:
                    csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
                
                if os.path.exists(csv_path):
                    output_files['features_csv'] = csv_path
                    try:
                        output_files['features_df'] = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Warning: Could not load existing CSV: {e}")
            
            # Add GeoJSON paths if they exist
            if config.save_geojson:
                if config.compartment_mode == "both":
                    cell_geojson = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
                    nuclear_geojson = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
                    if os.path.exists(cell_geojson):
                        output_files['cell_geojson'] = cell_geojson
                    if os.path.exists(nuclear_geojson):
                        output_files['nuclear_geojson'] = nuclear_geojson
                    # Primary GeoJSON
                    primary_geojson = cell_geojson if config.primary_output_mask == "cell" else nuclear_geojson
                    if os.path.exists(primary_geojson):
                        output_files['geojson'] = primary_geojson
                else:
                    if config.compartment_mode == "nuclear":
                        geojson_path = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
                    else:
                        geojson_path = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
                    
                    if os.path.exists(geojson_path):
                        output_files['geojson'] = geojson_path
            
            return output_files
    
    print(f"Processing image: {input_image_name}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract channel information
    channel_names = get_channel_names(ome_tiff_path)
    print(f"Channel names: {channel_names}")
    
    try:
        dapi_index = channel_names.index(dapi_channel)
    except ValueError:
        raise ValueError(f"'{dapi_channel}' channel not found in image")
    
    # Determine marker indices for membrane signal
    if markers is None:
        # Use all channels except DAPI
        marker_indices = [i for i, name in enumerate(channel_names) if name != dapi_channel]
    else:
        marker_indices = [channel_names.index(marker) for marker in markers if marker in channel_names]
    
    print(f"DAPI channel index: {dapi_index}")
    print(f"Marker channel indices: {marker_indices}")
    
    # Load image data
    with tiff.TiffFile(ome_tiff_path) as tif:
        dapi_image = tif.pages[dapi_index].asarray().astype(np.float32)
        marker_images = [tif.pages[i].asarray().astype(np.float32) for i in marker_indices]
        all_images = [tif.pages[i].asarray() for i in range(len(channel_names))]
    
    # Create membrane signal
    if config.merge_membrane_markers and marker_images:
        print(f"Merging {len(marker_images)} markers using max projection")
        membrane_image = np.max(marker_images, axis=0)
    elif marker_images:
        print(f"Using first available marker: {channel_names[marker_indices[0]]}")
        membrane_image = marker_images[0]
    else:
        # Fallback to a non-DAPI channel
        fallback_index = 1 if dapi_index != 1 else 2
        print(f"No markers found, using fallback channel: {channel_names[fallback_index]}")
        with tiff.TiffFile(ome_tiff_path) as tif:
            membrane_image = tif.pages[fallback_index].asarray().astype(np.float32)
    
    # Apply tissue mask if specified
    if config.use_input_geojson and os.path.exists(config.input_geojson_path):
        print("Applying tissue mask from GeoJSON...")
        tissue_mask = create_tissue_mask(dapi_image, config.input_geojson_path)
        membrane_image = membrane_image * tissue_mask
        dapi_image = dapi_image * tissue_mask
    
    # Normalize images
    dapi_min, dapi_max = float(dapi_image.min()), float(dapi_image.max())
    dapi_range = max(dapi_max - dapi_min, 1e-6)
    dapi_norm = (dapi_image - dapi_min) / dapi_range
    
    membrane_min, membrane_max = float(membrane_image.min()), float(membrane_image.max())
    membrane_range = max(membrane_max - membrane_min, 1e-6)
    membrane_norm = (membrane_image - membrane_min) / membrane_range
    
    # Prepare image stack for segmentation (nuclear first, membrane second)
    image_for_segmentation = np.stack([dapi_norm, membrane_norm], axis=-1)
    
    print(f"DeepCell input shape: {image_for_segmentation.shape}")
    
    # Initialize segmenter and run segmentation
    segmenter = DeepCellSegmenter(config)
    seg_results = segmenter.segment(image_for_segmentation)
    
    # Extract features from segmentation results
    print("Extracting cell features...")
    output_files = {}
    
    # Handle different compartment modes
    if config.compartment_mode == "both":
        # Extract features for both cell and nuclear masks
        cell_mask = seg_results['cell']
        nuclear_mask = seg_results['nuclear']
        
        # Cell features
        cell_features_df = extract_features(cell_mask, all_images, channel_names)
        cell_csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
        cell_features_df.to_csv(cell_csv_path, index=False)
        output_files['cell_features_csv'] = cell_csv_path
        print(f"Cell features saved to: {cell_csv_path}")
        
        # Nuclear features
        nuclear_features_df = extract_features(nuclear_mask, all_images, channel_names)
        nuclear_csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
        nuclear_features_df.to_csv(nuclear_csv_path, index=False)
        output_files['nuclear_features_csv'] = nuclear_csv_path
        print(f"Nuclear features saved to: {nuclear_csv_path}")
        
        # Set primary features based on configuration
        if config.primary_output_mask == "nuclear":
            output_files['features_df'] = nuclear_features_df
            output_files['features_csv'] = nuclear_csv_path
            output_files['masks'] = nuclear_mask
        else:
            output_files['features_df'] = cell_features_df
            output_files['features_csv'] = cell_csv_path
            output_files['masks'] = cell_mask
        
        # Store both masks
        output_files['cell_mask'] = cell_mask
        output_files['nuclear_mask'] = nuclear_mask
        
    else:
        # Single compartment mode
        primary_mask = seg_results['primary']
        features_df = extract_features(primary_mask, all_images, channel_names)
        
        # Determine filename based on compartment mode
        if config.compartment_mode == "nuclear":
            csv_path = os.path.join(output_dir, f"{input_image_name}_nuclear_features.csv")
        else:
            csv_path = os.path.join(output_dir, f"{input_image_name}_cell_features.csv")
        
        features_df.to_csv(csv_path, index=False)
        output_files['features_csv'] = csv_path
        output_files['features_df'] = features_df
        output_files['masks'] = primary_mask
        print(f"Features saved to: {csv_path}")
    
    # Save segmentation masks as NPY
    if config.save_npy:
        if config.compartment_mode == "both":
            cell_npy_path = os.path.join(output_dir, f"{input_image_name}_cell_mask.npy")
            nuclear_npy_path = os.path.join(output_dir, f"{input_image_name}_nuclear_mask.npy")
            np.save(cell_npy_path, seg_results['cell'].astype(np.uint16), allow_pickle=False)
            np.save(nuclear_npy_path, seg_results['nuclear'].astype(np.uint16), allow_pickle=False)
            output_files['cell_mask_npy'] = cell_npy_path
            output_files['nuclear_mask_npy'] = nuclear_npy_path
            print(f"Masks saved as NPY files")
        else:
            if config.compartment_mode == "nuclear":
                npy_path = os.path.join(output_dir, f"{input_image_name}_nuclear_mask.npy")
            else:
                npy_path = os.path.join(output_dir, f"{input_image_name}_cell_mask.npy")
            
            np.save(npy_path, seg_results['primary'].astype(np.uint16), allow_pickle=False)
            output_files['mask_npy'] = npy_path
            print(f"Mask saved to: {npy_path}")
    
    # Save cell boundaries as GeoJSON
    if config.save_geojson:
        if config.compartment_mode == "both":
            # Save both cell and nuclear boundaries
            cell_geojson_path = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
            nuclear_geojson_path = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
            
            generate_geojson_from_mask(seg_results['cell'], cell_geojson_path, "cell")
            generate_geojson_from_mask(seg_results['nuclear'], nuclear_geojson_path, "nuclear")
            
            output_files['cell_geojson'] = cell_geojson_path
            output_files['nuclear_geojson'] = nuclear_geojson_path
            
            # Set primary GeoJSON
            if config.primary_output_mask == "nuclear":
                output_files['geojson'] = nuclear_geojson_path
            else:
                output_files['geojson'] = cell_geojson_path
        else:
            # Single compartment mode
            if config.compartment_mode == "nuclear":
                geojson_path = os.path.join(output_dir, f"{input_image_name}_nuclear_boundaries.geojson")
                mask_type = "nuclear"
            else:
                geojson_path = os.path.join(output_dir, f"{input_image_name}_cell_boundaries.geojson")
                mask_type = "cell"
            
            generate_geojson_from_mask(seg_results['primary'], geojson_path, mask_type)
            output_files['geojson'] = geojson_path
    
    # Store segmentation results
    output_files['seg_results'] = seg_results
    output_files['styles'] = None  # DeepCell doesn't have styles like Cellpose
    
    elapsed_time = datetime.now() - start_time
    print(f"Segmentation completed in: {elapsed_time}")
    
    return output_files


def batch_segment_images(input_dir, output_dir, config, file_pattern="*.ome.tiff", 
                        markers=None, dapi_channel='DAPI', overwrite=False):
    """
    Batch process multiple OME-TIFF images for segmentation.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input OME-TIFF files
    output_dir : str
        Directory to save outputs
    config : DeepCellConfig
        Configuration object
    file_pattern : str
        File pattern to match (default: "*.ome.tiff")
    markers : list of str, optional
        List of marker channel names to use
    dapi_channel : str
        Name of DAPI channel
    overwrite : bool
        Whether to overwrite existing outputs (default: False)
        
    Returns:
    --------
    dict
        Dictionary mapping image names to their output files
    """
    import glob
    
    # Find all matching files
    search_pattern = os.path.join(input_dir, file_pattern)
    image_files = glob.glob(search_pattern)
    
    if not image_files:
        print(f"No files found matching pattern: {search_pattern}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    
    results = {}
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n--- Processing image {i}/{len(image_files)} ---")
        
        try:
            output_files = segment_image_deepcell(
                image_path, output_dir, config, markers, dapi_channel, overwrite
            )
            results[os.path.basename(image_path)] = output_files
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[os.path.basename(image_path)] = {'error': str(e)}
    
    print(f"\nBatch processing completed. Processed {len(results)} images.")
    return results


# Placeholder for future segmentation methods
class OtherSegmenter:
    """Placeholder for other segmentation methods (future implementation)."""
    
    def __init__(self, config):
        self.config = config
        raise NotImplementedError("Other segmentation methods not yet implemented")
    
    def segment(self, image_stack):
        pass
