"""
Utility functions for the spatial annotation pipeline.

This module contains the refactored and consolidated logic from the original
scripts for GeoJSON processing, cell annotation, and tumor proximity analysis.
"""
import os
import json
import glob
import uuid
import pandas as pd
from shapely.geometry import shape, Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid
from scipy.spatial import cKDTree
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)

# Constants for pixel size conversion
PIXEL_SIZE_X = 0.5083  # µm per pixel
PIXEL_SIZE_Y = 0.5083  # µm per pixel
PIXEL_SIZE_UM = 0.5083  # Average pixel size in micrometers
PIXEL_SIZE_MM = PIXEL_SIZE_UM / 1000  # Convert to millimeters
AREA_CONVERSION_FACTOR = PIXEL_SIZE_UM ** 2  # Convert pixel area to µm²

def merge_polygons(polygons):
    """
    If there are no polygons, return None.
    If there's exactly one polygon, return it.
    If multiple, return a unary_union of them.
    """
    if len(polygons) == 0:
        return None
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return unary_union(polygons)

def split_sort_polygons(geom):
    """
    Given a (Multi)Polygon geometry, split into individual polygons,
    sort them by area (descending), and return a list of dicts:
    [
      {
        "id": <int>,
        "geom": poly,
        "prepped": prep(poly),
        "bounds": (minx, miny, maxx, maxy),
        "area_um2": <float>
      },
      ...
    ]
    The largest polygon gets id=1, the second largest id=2, etc.
    """
    if geom is None:
        return []

    if geom.geom_type == "Polygon":
        splitted = [geom]
    elif geom.geom_type == "MultiPolygon":
        splitted = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        # Extract only Polygon and MultiPolygon objects from the collection
        splitted = []
        for sub_geom in geom.geoms:
            if sub_geom.geom_type == "Polygon":
                splitted.append(sub_geom)
            elif sub_geom.geom_type == "MultiPolygon":
                splitted.extend(list(sub_geom.geoms))
    else:
        splitted = []

    # Sort polygons by area desc
    splitted_sorted = sorted(splitted, key=lambda p: p.area, reverse=True)

    poly_list = []
    for idx, poly in enumerate(splitted_sorted, start=1):
        poly_prepped = prep(poly)
        minx, miny, maxx, maxy = poly.bounds
        pix_area = poly.area  # area in pixel^2
        um2_area = pix_area * AREA_CONVERSION_FACTOR

        poly_list.append({
            "id": idx,
            "geom": poly,
            "prepped": poly_prepped,
            "bounds": (minx, miny, maxx, maxy),
            "area_um2": um2_area
        })

    return poly_list

def classify_point_in_polygon_lists(x, y, poly_list):
    """
    Check a point (x, y) against a list of polygons
    in descending area order. Return (poly_id, poly_area_um2) for
    the first match. If no match, return (0, 0.0).
    """
    pt = Point(x, y)
    for poly_info in poly_list:
        minx, miny, maxx, maxy = poly_info["bounds"]
        if (minx <= x <= maxx) and (miny <= y <= maxy):
            if poly_info["prepped"].contains(pt):
                return (poly_info["id"], poly_info["area_um2"])
    return (0, 0.0)

def annotate_csv_with_regions_and_zones(csv_path, geojson_path, output_path):
    """
    Annotate cells in CSV with tissue regions, zones, and tumor proximity.
    Includes ROI tracking (tissue_roi, tissue_roi_size, etc.) like reference code.
    
    Args:
        csv_path: Path to segmentation CSV file
        geojson_path: Path to processed GeoJSON file with compartments and zones
        output_path: Path for output annotated CSV
        
    Returns:
        Number of cells annotated
    """
    import json
    from shapely.geometry import shape
    
    logger.info(f"Annotating cells from {os.path.basename(csv_path)}")
    
    # Load segmentation data
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"Empty CSV file: {csv_path}")
        return 0
        
    logger.info(f"Loaded {len(df):,} cells")
    
    # Load GeoJSON annotations
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    features = geojson_data.get('features', [])
    logger.info(f"Loaded {len(features)} annotation features")
    
    # Organize features by type for ROI tracking
    tissue_polygons = []
    epithelia_polygons = []
    stroma_polygons = []
    tumor_polygons = []
    endothelia_polygons = []
    
    zone_a_polygons = []
    zone_b_polygons = []
    zone_c_polygons = []
    zone_d_polygons = []
    zone_aa_polygons = []
    zone_bb_polygons = []
    zone_cc_polygons = []
    zone_dd_polygons = []
    
    for feature in features:
        name = feature.get('properties', {}).get('classification', {}).get('name', '').lower()
        geom = shape(feature.get('geometry', {}))
        
        if name == 'tissue':
            tissue_polygons.append(geom)
        elif name == 'epithelia':
            epithelia_polygons.append(geom)
        elif name == 'stroma':
            stroma_polygons.append(geom)
        elif name == 'tumor':
            tumor_polygons.append(geom)
        elif name == 'endothelia':
            endothelia_polygons.append(geom)
        elif name == 'zone_a':
            zone_a_polygons.append(geom)
        elif name == 'zone_b':
            zone_b_polygons.append(geom)
        elif name == 'zone_c':
            zone_c_polygons.append(geom)
        elif name == 'zone_d':
            zone_d_polygons.append(geom)
        elif name == 'zone_aa':
            zone_aa_polygons.append(geom)
        elif name == 'zone_bb':
            zone_bb_polygons.append(geom)
        elif name == 'zone_cc':
            zone_cc_polygons.append(geom)
        elif name == 'zone_dd':
            zone_dd_polygons.append(geom)
    
    logger.info(f"Found {len(tissue_polygons)} tissue, {len(epithelia_polygons)} epithelia, {len(stroma_polygons)} stroma, {len(tumor_polygons)} tumor, {len(endothelia_polygons)} endothelia polygons")
    logger.info(f"Found {len(zone_a_polygons)} zone_a, {len(zone_b_polygons)} zone_b, {len(zone_c_polygons)} zone_c, {len(zone_d_polygons)} zone_d polygons")
    logger.info(f"Found {len(zone_aa_polygons)} zone_aa, {len(zone_bb_polygons)} zone_bb, {len(zone_cc_polygons)} zone_cc, {len(zone_dd_polygons)} zone_dd polygons")
    
    # Create merged and split polygons for ROI tracking
    logger.info("Merging and splitting polygons for ROI tracking...")
    
    merged_tissue = merge_polygons(tissue_polygons)
    merged_epithelia = merge_polygons(epithelia_polygons)
    merged_stroma = merge_polygons(stroma_polygons)
    merged_tumor = merge_polygons(tumor_polygons)
    merged_endothelia = merge_polygons(endothelia_polygons)
    
    merged_zone_a = merge_polygons(zone_a_polygons)
    merged_zone_b = merge_polygons(zone_b_polygons)
    merged_zone_c = merge_polygons(zone_c_polygons)
    merged_zone_d = merge_polygons(zone_d_polygons)
    merged_zone_aa = merge_polygons(zone_aa_polygons)
    merged_zone_bb = merge_polygons(zone_bb_polygons)
    merged_zone_cc = merge_polygons(zone_cc_polygons)
    merged_zone_dd = merge_polygons(zone_dd_polygons)
    
    # Split and sort each merged geometry
    tissue_poly_list = split_sort_polygons(merged_tissue)
    epithelia_poly_list = split_sort_polygons(merged_epithelia)
    stroma_poly_list = split_sort_polygons(merged_stroma)
    tumor_poly_list = split_sort_polygons(merged_tumor)
    endothelia_poly_list = split_sort_polygons(merged_endothelia)
    
    zone_a_poly_list = split_sort_polygons(merged_zone_a)
    zone_b_poly_list = split_sort_polygons(merged_zone_b)
    zone_c_poly_list = split_sort_polygons(merged_zone_c)
    zone_d_poly_list = split_sort_polygons(merged_zone_d)
    zone_aa_poly_list = split_sort_polygons(merged_zone_aa)
    zone_bb_poly_list = split_sort_polygons(merged_zone_bb)
    zone_cc_poly_list = split_sort_polygons(merged_zone_cc)
    zone_dd_poly_list = split_sort_polygons(merged_zone_dd)
    
    logger.info(f"Split into: Tissue={len(tissue_poly_list)}, Epithelia={len(epithelia_poly_list)}, Stroma={len(stroma_poly_list)}, Tumor={len(tumor_poly_list)}, Endothelia={len(endothelia_poly_list)} individual polygons")
    
    # Get cell coordinates (try different column name variations)
    x_col = None
    y_col = None
    
    for col in df.columns:
        if 'x' in col.lower() and ('centroid' in col.lower() or 'center' in col.lower()):
            x_col = col
        elif 'y' in col.lower() and ('centroid' in col.lower() or 'center' in col.lower()):
            y_col = col
    
    if x_col is None or y_col is None:
        # Fallback to common naming conventions
        potential_x = [c for c in df.columns if c.lower() in ['x', 'x_centroid', 'centroid_x', 'center_x']]
        potential_y = [c for c in df.columns if c.lower() in ['y', 'y_centroid', 'centroid_y', 'center_y']]
        
        if potential_x:
            x_col = potential_x[0]
        if potential_y:
            y_col = potential_y[0]
    
    if x_col is None or y_col is None:
        logger.error(f"Could not find X/Y coordinate columns in CSV. Available columns: {list(df.columns)}")
        return 0
        
    logger.info(f"Using coordinates: {x_col}, {y_col}")
    
    # Initialize annotation columns (binary flags + ROI info like reference code)
    df['tissue'] = 0
    df['epithelia'] = 0
    df['stroma'] = 0
    df['tumor'] = 0
    df['endothelia'] = 0
    
    # ROI tracking columns
    df['tissue_roi'] = 0
    df['tissue_roi_size'] = 0.0
    df['epithelia_roi'] = 0
    df['epithelia_roi_size'] = 0.0
    df['stroma_roi'] = 0
    df['stroma_roi_size'] = 0.0
    df['tumor_roi'] = 0
    df['tumor_roi_size'] = 0.0
    df['endothelia_roi'] = 0
    df['endothelia_roi_size'] = 0.0
    
    # Zone binary flag columns
    df['zone_a'] = 0
    df['zone_b'] = 0
    df['zone_c'] = 0
    df['zone_d'] = 0
    df['zone_aa'] = 0
    df['zone_bb'] = 0
    df['zone_cc'] = 0
    df['zone_dd'] = 0
    
    # Zone-T columns (tumor proximity)
    df['zone_a_t'] = 0
    df['zone_b_t'] = 0
    df['zone_c_t'] = 0
    df['zone_d_t'] = 0
    
    # Annotate each cell using the reference code approach
    logger.info("Classifying each cell with ROI tracking...")
    total_cells = len(df)
    
    for i, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        
        # Tissue classification with ROI tracking
        t_id, t_area = classify_point_in_polygon_lists(x, y, tissue_poly_list)
        if t_id > 0:
            df.at[i, 'tissue'] = 1
        df.at[i, 'tissue_roi'] = t_id
        df.at[i, 'tissue_roi_size'] = t_area
        
        # Epithelia classification with ROI tracking
        e_id, e_area = classify_point_in_polygon_lists(x, y, epithelia_poly_list)
        if e_id > 0:
            df.at[i, 'epithelia'] = 1
        df.at[i, 'epithelia_roi'] = e_id
        df.at[i, 'epithelia_roi_size'] = e_area
        
        # Stroma classification with ROI tracking
        s_id, s_area = classify_point_in_polygon_lists(x, y, stroma_poly_list)
        if s_id > 0:
            df.at[i, 'stroma'] = 1
        df.at[i, 'stroma_roi'] = s_id
        df.at[i, 'stroma_roi_size'] = s_area
        
        # Tumor classification with ROI tracking
        tm_id, tm_area = classify_point_in_polygon_lists(x, y, tumor_poly_list)
        if tm_id > 0:
            df.at[i, 'tumor'] = 1
        df.at[i, 'tumor_roi'] = tm_id
        df.at[i, 'tumor_roi_size'] = tm_area
        
        # Endothelia classification with ROI tracking
        en_id, en_area = classify_point_in_polygon_lists(x, y, endothelia_poly_list)
        if en_id > 0:
            df.at[i, 'endothelia'] = 1
        df.at[i, 'endothelia_roi'] = en_id
        df.at[i, 'endothelia_roi_size'] = en_area
        
        # Zones (binary flags only like reference code)
        za_id, _ = classify_point_in_polygon_lists(x, y, zone_a_poly_list)
        if za_id > 0:
            df.at[i, 'zone_a'] = 1
            
        zb_id, _ = classify_point_in_polygon_lists(x, y, zone_b_poly_list)
        if zb_id > 0:
            df.at[i, 'zone_b'] = 1
            
        zc_id, _ = classify_point_in_polygon_lists(x, y, zone_c_poly_list)
        if zc_id > 0:
            df.at[i, 'zone_c'] = 1
            
        zd_id, _ = classify_point_in_polygon_lists(x, y, zone_d_poly_list)
        if zd_id > 0:
            df.at[i, 'zone_d'] = 1
            
        zaa_id, _ = classify_point_in_polygon_lists(x, y, zone_aa_poly_list)
        if zaa_id > 0:
            df.at[i, 'zone_aa'] = 1
            
        zbb_id, _ = classify_point_in_polygon_lists(x, y, zone_bb_poly_list)
        if zbb_id > 0:
            df.at[i, 'zone_bb'] = 1
            
        zcc_id, _ = classify_point_in_polygon_lists(x, y, zone_cc_poly_list)
        if zcc_id > 0:
            df.at[i, 'zone_cc'] = 1
            
        zdd_id, _ = classify_point_in_polygon_lists(x, y, zone_dd_poly_list)
        if zdd_id > 0:
            df.at[i, 'zone_dd'] = 1
        
        # Progress reporting
        if (i + 1) % 5000 == 0 or i == total_cells - 1:
            progress = (i + 1) / total_cells * 100
            logger.info(f"Progress: {progress:.1f}% ({i+1}/{total_cells})")

    # Calculate tumor proximity for stromal zones (Zone-T) using cKDTree
    logger.info("Computing tumor proximity (Zone-T) using cKDTree...")
    
    coords = df[[x_col, y_col]].values
    tumor_coords = df.loc[df['tumor'] == 1, [x_col, y_col]].values
    epi_coords = df.loc[df['epithelia'] == 1, [x_col, y_col]].values

    for zone in ['zone_a', 'zone_b', 'zone_c', 'zone_d']:
        zone_t_col = f'{zone}_t'
        
        # Find indices of cells within the current zone
        idx_zone = df[zone] == 1
        if not idx_zone.any():
            continue # Skip if no cells are in this zone

        zone_coords = coords[idx_zone]

        # Handle edge cases
        if tumor_coords.shape[0] == 0:
            # No tumor cells, all Zone-T are 0 (closer to epithelia)
            df.loc[idx_zone, zone_t_col] = 0
        elif epi_coords.shape[0] == 0:
            # No epithelia cells, all Zone-T are 1 (closer to tumor)
            df.loc[idx_zone, zone_t_col] = 1
        else:
            # Both tumor and epithelia are present, calculate distances
            tumor_tree = cKDTree(tumor_coords)
            epi_tree = cKDTree(epi_coords)
            
            tumor_dist, _ = tumor_tree.query(zone_coords, k=1)
            epi_dist, _ = epi_tree.query(zone_coords, k=1)
            
            closer_to_tumor = tumor_dist < epi_dist
            df.loc[idx_zone, zone_t_col] = closer_to_tumor.astype(int)
    
    # Save annotated CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved annotated data: {len(df):,} cells with spatial annotations and ROI tracking")
    
    return len(df)

def process_all_csvs_stage2(csv_dir, geojson_dir, output_dir, csv_pattern="*_features.csv", overwrite=False):
    """
    Process all CSV files for Stage 2 annotation.
    
    Args:
        csv_dir: Directory containing segmentation CSV files
        geojson_dir: Directory containing processed GeoJSON files from Stage 1B
        output_dir: Directory for output annotated CSV files
        csv_pattern: Pattern to match CSV files
        overwrite: Whether to overwrite existing output files
        
    Returns:
        Dictionary with processing results
    """
    import glob
    import os
    
    logger.info("Starting Stage 2 batch processing...")
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(csv_dir, csv_pattern))
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    if not csv_files:
        logger.warning(f"No CSV files found matching pattern {csv_pattern} in {csv_dir}")
        return {}
    
    # Find corresponding GeoJSON files
    geojson_files = glob.glob(os.path.join(geojson_dir, "*_final_annotations.geojson"))
    logger.info(f"Found {len(geojson_files)} GeoJSON files for annotation")
    
    if not geojson_files:
        logger.warning(f"No GeoJSON files found in {geojson_dir}")
        return {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for csv_file in csv_files:
        csv_basename = os.path.basename(csv_file)
        # Extract slide name from CSV filename - handle multiple patterns
        slide_name = csv_basename.replace('_features.csv', '').replace('.csv', '')
        
        # Try to extract the core slide identifier (e.g., "P11_roi_1" from "P11_roi_1.ome_cell_features")
        if '_cell_features' in slide_name:
            core_slide_name = slide_name.replace('_cell_features', '').replace('.ome', '')
        elif '.ome' in slide_name:
            core_slide_name = slide_name.split('.ome')[0]
        else:
            core_slide_name = slide_name
        
        logger.info(f"\nProcessing slide: {slide_name}")
        logger.info(f"  Core slide name: {core_slide_name}")
        logger.info(f"  CSV: {csv_basename}")
        
        # Find matching GeoJSON file - try multiple matching strategies
        matching_geojson = None
        for geojson_file in geojson_files:
            geojson_basename = os.path.basename(geojson_file)
            
            # Strategy 1: Direct match with slide_name
            if slide_name in geojson_basename or geojson_basename.startswith(slide_name):
                matching_geojson = geojson_file
                break
            
            # Strategy 2: Match with core_slide_name
            if core_slide_name in geojson_basename or geojson_basename.startswith(core_slide_name):
                matching_geojson = geojson_file
                break
            
            # Strategy 3: Try reverse match (geojson name in slide name)
            geojson_core = geojson_basename.replace('_final_annotations.geojson', '')
            if geojson_core in slide_name or slide_name.startswith(geojson_core):
                matching_geojson = geojson_file
                break
                matching_geojson = geojson_file
                break
        
        if not matching_geojson:
            logger.warning(f"  No matching GeoJSON found for {slide_name}")
            results[slide_name] = {'success': False, 'error': 'No matching GeoJSON file'}
            continue
        
        logger.info(f"  GeoJSON: {os.path.basename(matching_geojson)}")
        
        # Generate output filename - preserve exact input name and add _annotated suffix
        input_filename = os.path.basename(csv_file)
        if input_filename.endswith('.csv'):
            output_filename = input_filename.replace('.csv', '_annotated.csv')
        else:
            output_filename = f"{input_filename}_annotated.csv"
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"  Output: {output_filename}")
        
        # Check if output already exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            logger.info(f"  Skipping {slide_name} - output file already exists (overwrite=False)")
            results[slide_name] = {
                'success': True,
                'skipped': True,
                'reason': 'Output file already exists',
                'output_file': output_path
            }
            continue
        
        try:
            cell_count = annotate_csv_with_regions_and_zones(
                csv_path=csv_file,
                geojson_path=matching_geojson,
                output_path=output_path
            )
            
            results[slide_name] = {
                'success': True,
                'cell_count': cell_count,
                'output_file': output_path
            }
            logger.info(f"  Success: {cell_count:,} cells annotated")
            
        except Exception as e:
            logger.error(f"  Error processing {slide_name}: {e}")
            results[slide_name] = {'success': False, 'error': str(e)}
    
    successful = sum(1 for r in results.values() if r.get('success'))
    logger.info(f"\nStage 2 processing complete: {successful}/{len(results)} files successful")
    
    return results

# Constants for pixel size and area conversion
PIXEL_SIZE_UM = 0.65  # pixel size in micrometers
PIXEL_SIZE_MM = PIXEL_SIZE_UM * 1e-3  # convert to mm
AREA_CONVERSION_FACTOR = PIXEL_SIZE_MM**2  # convert pixel^2 => mm^2

# ======================================================================================
# STAGE 1A: MERGE AND CLEAN TISSUE (INITIAL PROCESSING)
# ======================================================================================

def merge_annotation_files(
    input_filepaths,
    output_geojson_path,
    tissue_classes=["epithelia", "tumor", "stroma"],
    endothelia_classes=["endothelial", "endothelia"],
    removal_classes=["artefact", "exclude", "blood_clot"],
    rename_map=None,
    min_area=300000,
    merged_class_name="tissue",
    output_options=None
):
    """
    Merge multiple annotation files into a single processed output.
    
    Args:
        input_filepaths: List of paths to GeoJSON files to merge
        output_geojson_path: Path to save the result
        ...
    """
    logger.info(f"Processing merge for {len(input_filepaths)} files -> {os.path.basename(output_geojson_path)}")
    
    if rename_map is None:
        rename_map = {}
    
    if output_options is None:
        output_options = {
            "include_merged": True,
            "include_components": True,
            "include_removal": False
        }

    # Collect polygons by type
    epithelia_polygons = []
    tumor_polygons = []
    stroma_polygons = []
    endothelia_polygons = []
    removal_polygons = []
    other_polygons = [] 
    
    # 1. Load ALL input files
    for f_path in input_filepaths:
        if not os.path.exists(f_path):
            logger.warning(f"Input file not found: {f_path}")
            continue
            
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                classification = props.get("classification", {})
                name = classification.get("name", "").lower()
                
                # Apply renaming
                if name in rename_map:
                    name = rename_map[name].lower()
                
                geom = shape(feature.get("geometry"))
                if not geom.is_valid:
                    geom = make_valid(geom)
                
                # Categorize
                if name in [c.lower() for c in tissue_classes]:
                    if "epithelia" in name: epithelia_polygons.append(geom)
                    elif "tumor" in name: tumor_polygons.append(geom)
                    elif "stroma" in name: stroma_polygons.append(geom)
                    else: stroma_polygons.append(geom)
                elif name in [c.lower() for c in endothelia_classes]:
                    endothelia_polygons.append(geom)
                elif name in [c.lower() for c in removal_classes]:
                    removal_polygons.append(geom)
                else:
                    other_polygons.append((name, geom))
                    
        except Exception as e:
            logger.error(f"Error reading {f_path}: {e}")

    # 2. Apply removal polygons
    if removal_polygons:
        removal_union = unary_union(removal_polygons)
        
        def remove_artifacts(poly_list):
            cleaned = []
            for poly in poly_list:
                try:
                    diff = poly.difference(removal_union)
                    if not diff.is_empty:
                        if diff.geom_type == 'MultiPolygon':
                            cleaned.extend([p for p in diff.geoms if not p.is_empty])
                        else:
                            cleaned.append(diff)
                except Exception as e:
                    logger.error(f"Error removing artifact: {e}")
                    cleaned.append(poly)
            return cleaned

        epithelia_polygons = remove_artifacts(epithelia_polygons)
        tumor_polygons = remove_artifacts(tumor_polygons)
        stroma_polygons = remove_artifacts(stroma_polygons)
        endothelia_polygons = remove_artifacts(endothelia_polygons)
        
        cleaned_other = []
        for name, poly in other_polygons:
             try:
                diff = poly.difference(removal_union)
                if not diff.is_empty:
                    if diff.geom_type == 'MultiPolygon':
                        for p in diff.geoms:
                            if not p.is_empty: cleaned_other.append((name, p))
                    else:
                        cleaned_other.append((name, diff))
             except:
                 cleaned_other.append((name, poly))
        other_polygons = cleaned_other
    
    # 3. Create merged tissue from all components
    all_tissue_components = epithelia_polygons + tumor_polygons + stroma_polygons + endothelia_polygons
    
    if not all_tissue_components:
        logger.warning("No tissue polygons found to merge")
        # Continue anyway to save components if they exist (e.g. only other polygons)
    
    logger.info(f"Creating merged tissue from {len(all_tissue_components)} total polygons")
    merged_tissue = unary_union(all_tissue_components)
    
    # 4. Break into individual tissue polygons and filter by size
    clean_tissue_polygons = []
    if merged_tissue is None:
        pass
    elif merged_tissue.is_empty:
        pass
    elif merged_tissue.geom_type == "Polygon":
        if merged_tissue.area >= min_area:
            clean_tissue_polygons.append(merged_tissue)
    elif merged_tissue.geom_type == "MultiPolygon":
        for poly in merged_tissue.geoms:
            if poly.area >= min_area:
                clean_tissue_polygons.append(poly)
    
    logger.info(f"Created {len(clean_tissue_polygons)} tissue polygons")
    
    # 5. Create output GeoJSON
    output_features = []
    
    # Helper to add features
    def add_features(polys, name, color):
        for poly in polys:
            if poly.is_empty: continue
            if poly.area < 1: continue 
            
            feat = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": {
                    "classification": {"name": name, "colorRGB": color},
                    "isLocked": False,
                    "measurements": []
                }
            }
            output_features.append(feat)

    # Add individual components if requested
    if output_options.get("include_components", True):
        add_features(epithelia_polygons, "epithelia", -16776961) # Blue
        add_features(tumor_polygons, "tumor", -65536)       # Red
        add_features(stroma_polygons, "stroma", -16711936)  # Green
        add_features(endothelia_polygons, "endothelia", -256) # Yellow
        
        for name, poly in other_polygons:
            add_features([poly], name, -7829368) # Grayish

    # Add removal polygons if requested
    if output_options.get("include_removal", False):
        add_features(removal_polygons, "Artifact", -16777216) # Black

    # Add merged tissue polygons
    if output_options.get("include_merged", True):
        for poly in clean_tissue_polygons:
            feat = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": {
                    "classification": {"name": merged_class_name, "colorRGB": -3355444}, # Light Gray
                    "isLocked": False,
                    "measurements": []
                }
            }
            output_features.append(feat)
    
    # Save output
    output_data = {
        "type": "FeatureCollection",
        "features": output_features
    }
    
    save_compact_geojson(output_data, output_geojson_path)
    
    return len(clean_tissue_polygons)

# Keep old function signature for compatibility if needed, or just remove it.
# Since we are updating the UI to use the new one, we can remove the old one or alias it.
# For safety, let's alias it but it won't work exactly the same way unless we map args.
# Actually, let's just replace it completely as we did above.

def merge_tissue_and_remove_artifacts(
    tissue_geojson_path,
    endothelia_geojson_path,
    removal_geojson_path,
    output_geojson_path,
    tissue_classes=["epithelia", "tumor", "stroma"],
    endothelia_classes=["endothelial", "endothelia"],
    removal_classes=["artefact", "exclude", "blood_clot"],
    rename_map=None,
    min_area=300000,
    merged_class_name="tissue",
    output_options=None
):
    """Wrapper for backward compatibility"""
    inputs = []
    if tissue_geojson_path:
        inputs.append(tissue_geojson_path)
    if endothelia_geojson_path and endothelia_geojson_path != tissue_geojson_path:
        inputs.append(endothelia_geojson_path)
    if removal_geojson_path and removal_geojson_path != tissue_geojson_path:
        if removal_geojson_path not in inputs:
            inputs.append(removal_geojson_path)
            
    return merge_annotation_files(
        input_filepaths=inputs,
        output_geojson_path=output_geojson_path,
        tissue_classes=tissue_classes,
        endothelia_classes=endothelia_classes,
        removal_classes=removal_classes,
        rename_map=rename_map,
        min_area=min_area,
        merged_class_name=merged_class_name,
        output_options=output_options
    )


# ======================================================================================
# BATCH PROCESSING FUNCTIONS
# ======================================================================================

def create_geojson_feature(geometry, class_name, color):
    """Create a GeoJSON feature with proper QuPath formatting"""
    try:
        return {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "properties": {
                "classification": {
                    "name": class_name,
                    "color": color
                },
                "isLocked": False,
                "measurements": []
            },
            "geometry": mapping(geometry)
        }
    except Exception as e:
        logger.warning(f"Error creating feature for {class_name}: {e}")
        return None


def process_all_slides_stage1a(
    raw_annotations_dir=None,
    endothelia_dir=None, 
    removal_dir=None,
    output_dir=None,
    tissue_classes=["epithelia", "tumor", "stroma"],
    endothelia_classes=["endothelial", "endothelia"],
    removal_classes=["artefact", "exclude", "blood_clot"],
    min_area=300000,
    rename_map=None
):
    """
    Process all slides for Stage 1A: Merge tissue components and remove artifacts.
    """
    logger.info("Starting Stage 1A batch processing...")
    
    raw_annotations_dir = raw_annotations_dir or "../data/annotations/stroma_epi_bkg"
    endothelia_dir = endothelia_dir or "../data/annotations/endothelia"
    removal_dir = removal_dir or "../data/annotations/removal_annotations"
    output_dir = output_dir or "../outputs/processed_annotations/stage1a_merged_tissue"

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Find all main annotation files
    pattern = os.path.join(raw_annotations_dir, "*.geojson")
    annotation_files = glob.glob(pattern)
    
    logger.info(f"Found {len(annotation_files)} annotation files to process")
    
    for annotation_file in annotation_files:
        slide_base = os.path.basename(annotation_file).replace(".geojson", "")
        logger.info(f"\nProcessing slide: {slide_base}")
        
        # Map filename to exclusion file (handle different naming patterns)
        exclusion_filename = slide_base
        if ".ome.tiff - Merged Image" in exclusion_filename:
            exclusion_filename = exclusion_filename.replace(".ome.tiff - Merged Image", "")
        exclusion_filename += ".geojson"
        
        # Define paths
        endothelia_file = os.path.join(endothelia_dir, os.path.basename(annotation_file))
        removal_file = os.path.join(removal_dir, exclusion_filename)
        
        # Generate output filename that preserves original name for QuPath workflow
        output_filename = f"{slide_base}_merged_tissue.geojson"
        output_file = os.path.join(output_dir, output_filename)
        
        logger.info(f"  Input: {os.path.basename(annotation_file)}")
        logger.info(f"  Endothelia: {os.path.basename(endothelia_file) if os.path.exists(endothelia_file) else 'Not found'}")
        logger.info(f"  Removal: {exclusion_filename if os.path.exists(removal_file) else 'Not found'}")
        logger.info(f"  Output: {output_filename}")
        
        try:
            result = merge_tissue_and_remove_artifacts(
                tissue_geojson_path=annotation_file,
                endothelia_geojson_path=endothelia_file,
                removal_geojson_path=removal_file,
                output_geojson_path=output_file,
                tissue_classes=tissue_classes,
                endothelia_classes=endothelia_classes,
                removal_classes=removal_classes,
                min_area=min_area,
                rename_map=rename_map
            )
            
            results[slide_base] = {
                'success': True,
                'output_path': output_file,
                'tissue_polygons': result,
                'input_file': annotation_file,
                'endothelia_file': endothelia_file if os.path.exists(endothelia_file) else None,
                'removal_file': removal_file if os.path.exists(removal_file) else None
            }
            
            logger.info(f"  Success: {result} tissue polygons created")
            
        except Exception as e:
            logger.error(f"  Error processing {slide_base}: {e}")
            results[slide_base] = {
                'success': False,
                'error': str(e),
                'input_file': annotation_file
            }
    
    successful = sum(1 for r in results.values() if r.get('success'))
    logger.info(f"\nStage 1A processing complete: {successful}/{len(annotation_files)} files processed successfully")
    
    return results


# ======================================================================================
# STAGE 1B: CREATE COMPARTMENTS AND ZONES (AFTER MANUAL QUPATH EDITING)
# ======================================================================================

def extract_polygons_simple(geom):
    """Extract individual polygons from geometry, like reference code."""
    if geom.is_empty:
        return []
    if isinstance(geom, (Polygon,)):
        return [geom]
    elif isinstance(geom, (MultiPolygon,)):
        return list(geom.geoms)
    else:
        return []


def create_compartments_and_zones(
    refined_tissue_geojson_path,
    output_geojson_path,
    tissue_classes=["epithelia", "tumor", "stroma"],
    endothelia_classes=["endothelial", "endothelia"],
    buffer_size=40,
    min_area=3000
):
    """
    Create final compartments and spatial zones with EXTREME performance optimization.
    Prioritizes speed over geometric precision for massive performance gains.
    """
    logger.info(f"Processing {os.path.basename(refined_tissue_geojson_path)}...")

    # 1. Load input file and separate features
    if not os.path.exists(refined_tissue_geojson_path):
        raise FileNotFoundError(f"Refined tissue GeoJSON not found: {refined_tissue_geojson_path}")
        
    with open(refined_tissue_geojson_path, 'r') as f:
        data = json.load(f)
    
    features = data.get("features", [])
    
    tissue_list, epi_list, tum_list, endo_list = [], [], [], []
    for feat in features:
        cls = feat.get("properties", {}).get("classification", {}).get("name", "").lower()
        geom = shape(feat.get("geometry", {}))
        if geom.is_empty: continue
        if not geom.is_valid: geom = make_valid(geom)

        if cls == "tissue":
            tissue_list.append(geom)
        elif cls == "epithelia":
            epi_list.append(geom)
        elif cls == "tumor":
            tum_list.append(geom)
        elif cls in endothelia_classes or cls == "endothelia":
            endo_list.append(geom)

    if not tissue_list:
        logger.error("No 'tissue' annotation found in the refined file. Cannot proceed.")
        return 0
    
    # 2. ULTRA-FAST COMPONENT CREATION - use simplify for speed
    logger.info("Creating tissue and component unions with aggressive simplification...")
    
    # Simplify all geometries first for massive speed gain
    simplified_tissue = [geom.simplify(2.0, preserve_topology=False) for geom in tissue_list]
    simplified_epi = [geom.simplify(1.0, preserve_topology=False) for geom in epi_list]
    simplified_tum = [geom.simplify(1.0, preserve_topology=False) for geom in tum_list]
    simplified_endo = [geom.simplify(1.0, preserve_topology=False) for geom in endo_list]
    
    tissue_boundary = unary_union(simplified_tissue)
    
    # Create final components - use simplified geometries
    final_epi = unary_union(simplified_epi).intersection(tissue_boundary) if simplified_epi else Polygon()
    final_tum = unary_union(simplified_tum).intersection(tissue_boundary) if simplified_tum else Polygon()
    final_endo = unary_union(simplified_endo).intersection(tissue_boundary) if simplified_endo else Polygon()
    
    # Stroma calculation
    epi_tum_union = unary_union([final_epi, final_tum]) if not (final_epi.is_empty and final_tum.is_empty) else Polygon()
    final_stroma = tissue_boundary.difference(epi_tum_union)
    
    # 3. ULTRA-FAST ZONE CREATION - minimal precision, maximum speed
    logger.info(f"Creating zones with MAXIMUM SPEED approach...")
    logger.info(f"Found {len(epi_list)} epithelia, {len(tum_list)} tumor polygons")
    logger.info(f"epi_tum_union area: {epi_tum_union.area:.0f} pixels")
    logger.info(f"final_stroma area: {final_stroma.area:.0f} pixels")

    # Get large polygons with aggressive simplification
    large_polys = []
    for g in simplified_epi + simplified_tum:
        for p in extract_polygons_simple(g):
            if p.area >= min_area:
                # Further simplify for zone creation
                simplified_p = p.simplify(3.0, preserve_topology=False)
                large_polys.append(simplified_p)

    logger.info(f"Found {len(large_polys)} large polygons (>= {min_area} pixels) for zone creation")

    if not large_polys:
        logger.warning("No source polygons for zones - creating empty zones. This means:")
        logger.warning("  - No epithelia or tumor polygons meet the minimum area threshold")
        logger.warning("  - Or all polygons were simplified away")
        logger.warning("  - Consider reducing min_area parameter")
        zoneA = zoneB = zoneC = zoneD = Polygon()
        zoneAA = zoneBB = zoneCC = zoneDD = Polygon()
    else:
        # FASTEST possible zone creation
        source_union = unary_union(large_polys)

        # DIAGNOSTIC: Check if buffer size is appropriate
        source_area = source_union.area
        buffer_ratio = (buffer_size * 4) / source_area if source_area > 0 else float('inf')
        logger.info(f"Source union area: {source_area:.0f} pixels")
        logger.info(f"Buffer size: {buffer_size} pixels, ratio: {buffer_ratio:.3f}")

        if buffer_ratio > 0.5:
            logger.warning(f"Buffer size ({buffer_size}) is very large relative to source area ({source_area:.0f})")
            logger.warning("This may cause inner zones to be empty. Consider reducing buffer_size")

        # Use VERY LOW resolution for speed (sacrifice precision for performance)
        buf_resolution = 2  # Even lower than before
        
        logger.info("Creating ultra-fast buffers...")
        buf1 = source_union.buffer(buffer_size, resolution=buf_resolution)
        buf2 = source_union.buffer(2 * buffer_size, resolution=buf_resolution)
        buf3 = source_union.buffer(3 * buffer_size, resolution=buf_resolution)

        logger.info("Computing zones with minimal precision...")
        # FASTEST zone computation - no fancy operations
        zoneA = buf1.difference(source_union)
        if not final_stroma.is_empty:
            zoneA = zoneA.intersection(final_stroma)
        
        zoneB = buf2.difference(buf1)
        if not final_stroma.is_empty:
            zoneB = zoneB.intersection(final_stroma)
        
        zoneC = buf3.difference(buf2)
        if not final_stroma.is_empty:
            zoneC = zoneC.intersection(final_stroma)
        
        zoneD = final_stroma.difference(buf3) if not final_stroma.is_empty else Polygon()
        
        # INSIDE ZONES - robust creation with fallback logic
        logger.info("Creating inner zones (zone_aa, zone_bb, zone_cc, zone_dd)...")

        try:
            # Check if source_union is valid for buffer operations
            if source_union.is_empty or not source_union.is_valid:
                logger.warning("Source union is empty or invalid - cannot create inner zones")
                zoneAA = zoneBB = zoneCC = zoneDD = Polygon()
            else:
                logger.info(f"Source union area: {source_union.area:.0f} pixels")

                # Create negative buffers with validation
                buf_neg1 = source_union.buffer(-buffer_size, resolution=buf_resolution)
                buf_neg2 = source_union.buffer(-2 * buffer_size, resolution=buf_resolution)
                buf_neg3 = source_union.buffer(-3 * buffer_size, resolution=buf_resolution)

                # Validate buffer results
                buf_neg1_valid = not buf_neg1.is_empty and buf_neg1.is_valid
                buf_neg2_valid = not buf_neg2.is_empty and buf_neg2.is_valid
                buf_neg3_valid = not buf_neg3.is_empty and buf_neg3.is_valid

                logger.info(f"Negative buffer validation: buf_neg1={buf_neg1_valid}, buf_neg2={buf_neg2_valid}, buf_neg3={buf_neg3_valid}")
                logger.info(f"Negative buffer areas: buf_neg1={buf_neg1.area:.0f}, buf_neg2={buf_neg2.area:.0f}, buf_neg3={buf_neg3.area:.0f}")

                # Create inner zones with robust logic
                if buf_neg1_valid:
                    zoneAA = source_union.difference(buf_neg1)
                    logger.info(f"Zone AA created: {zoneAA.area:.0f} pixels")
                else:
                    # If buf_neg1 is invalid, zoneAA gets the entire source
                    zoneAA = source_union
                    logger.info(f"Zone AA fallback: using entire source ({zoneAA.area:.0f} pixels)")

                if buf_neg1_valid and buf_neg2_valid:
                    zoneBB = buf_neg1.difference(buf_neg2)
                    logger.info(f"Zone BB created: {zoneBB.area:.0f} pixels")
                elif buf_neg1_valid:
                    # If buf_neg2 is invalid but buf_neg1 is valid, zoneBB gets buf_neg1
                    zoneBB = buf_neg1
                    logger.info(f"Zone BB fallback: using buf_neg1 ({zoneBB.area:.0f} pixels)")
                else:
                    zoneBB = Polygon()
                    logger.info("Zone BB: empty (no valid buffers)")

                if buf_neg2_valid and buf_neg3_valid:
                    zoneCC = buf_neg2.difference(buf_neg3)
                    logger.info(f"Zone CC created: {zoneCC.area:.0f} pixels")
                elif buf_neg2_valid:
                    # If buf_neg3 is invalid but buf_neg2 is valid, zoneCC gets buf_neg2
                    zoneCC = buf_neg2
                    logger.info(f"Zone CC fallback: using buf_neg2 ({zoneCC.area:.0f} pixels)")
                else:
                    zoneCC = Polygon()
                    logger.info("Zone CC: empty (no valid buffers)")

                if buf_neg3_valid:
                    zoneDD = buf_neg3
                    logger.info(f"Zone DD created: {zoneDD.area:.0f} pixels")
                else:
                    zoneDD = Polygon()
                    logger.info("Zone DD: empty (buf_neg3 invalid)")

                logger.info(f"Inner zones before cropping: AA={zoneAA.area:.0f}, BB={zoneBB.area:.0f}, CC={zoneCC.area:.0f}, DD={zoneDD.area:.0f}")

                # Crop to epithelial/tumor areas with robust error handling
                if not epi_tum_union.is_empty:
                    logger.info(f"Cropping inner zones to epi_tum_union (area: {epi_tum_union.area:.0f})")

                    def safe_intersection(zone, name):
                        """Safely perform intersection with fallback options."""
                        if zone.is_empty:
                            return zone

                        try:
                            # First try: direct intersection
                            result = zone.intersection(epi_tum_union)
                            logger.info(f"Zone {name} after intersection: {result.area:.0f}")
                            return result
                        except Exception as e1:
                            logger.warning(f"Zone {name} direct intersection failed: {e1}")
                            try:
                                # Second try: buffer both geometries to fix issues
                                zone_fixed = zone.buffer(0.1).buffer(-0.1)
                                epi_fixed = epi_tum_union.buffer(0.1).buffer(-0.1)
                                result = zone_fixed.intersection(epi_fixed)
                                logger.info(f"Zone {name} after fixed intersection: {result.area:.0f}")
                                return result
                            except Exception as e2:
                                logger.warning(f"Zone {name} fixed intersection failed: {e2}")
                                # Third try: use the original zone if intersection fails
                                logger.warning(f"Zone {name} keeping original geometry")
                                return zone

                    zoneAA = safe_intersection(zoneAA, "AA")
                    zoneBB = safe_intersection(zoneBB, "BB")
                    zoneCC = safe_intersection(zoneCC, "CC")
                    zoneDD = safe_intersection(zoneDD, "DD")

                    logger.info(f"Inner zones after cropping: AA={zoneAA.area:.0f}, BB={zoneBB.area:.0f}, CC={zoneCC.area:.0f}, DD={zoneDD.area:.0f}")
                else:
                    logger.warning("epi_tum_union is empty - inner zones will be empty after cropping")
                    zoneAA = zoneBB = zoneCC = zoneDD = Polygon()

            logger.info(f"Final inner zone areas: AA={zoneAA.area:.0f}, BB={zoneBB.area:.0f}, CC={zoneCC.area:.0f}, DD={zoneDD.area:.0f}")

        except Exception as e:
            logger.warning(f"Inner zone creation failed: {e}, using empty zones")
            logger.warning(f"Exception details: {type(e).__name__}: {str(e)}")
            zoneAA = zoneBB = zoneCC = zoneDD = Polygon()
    
    logger.info(f"Zone areas - A: {zoneA.area:.0f}, B: {zoneB.area:.0f}, C: {zoneC.area:.0f}, D: {zoneD.area:.0f}")
    logger.info(f"Inner zone areas - AA: {zoneAA.area:.0f}, BB: {zoneBB.area:.0f}, CC: {zoneCC.area:.0f}, DD: {zoneDD.area:.0f}")

    # 4. Create final features for output
    output_features = []
    zone_counts = {'outer': 0, 'inner': 0, 'compartments': 0}
    
    # Add main compartments (only if not empty)
    if not tissue_boundary.is_empty:
        output_features.append(create_geojson_feature(tissue_boundary, "tissue", [0, 255, 0]))
        zone_counts['compartments'] += 1
    if not final_stroma.is_empty:
        output_features.append(create_geojson_feature(final_stroma, "stroma", [0, 0, 255]))
        zone_counts['compartments'] += 1
    if not final_epi.is_empty:
        output_features.append(create_geojson_feature(final_epi, "epithelia", [255, 0, 0]))
        zone_counts['compartments'] += 1
    if not final_tum.is_empty:
        output_features.append(create_geojson_feature(final_tum, "tumor", [255, 255, 0]))
        zone_counts['compartments'] += 1
    if not final_endo.is_empty:
        output_features.append(create_geojson_feature(final_endo, "endothelia", [255, 0, 255]))
        zone_counts['compartments'] += 1

    # Add zones (only if not empty)
    if not zoneA.is_empty:
        output_features.append(create_geojson_feature(zoneA, "zone_a", [255, 128, 128]))
        zone_counts['outer'] += 1
    if not zoneB.is_empty:
        output_features.append(create_geojson_feature(zoneB, "zone_b", [255, 160, 160]))
        zone_counts['outer'] += 1
    if not zoneC.is_empty:
        output_features.append(create_geojson_feature(zoneC, "zone_c", [255, 192, 192]))
        zone_counts['outer'] += 1
    if not zoneD.is_empty:
        output_features.append(create_geojson_feature(zoneD, "zone_d", [224, 224, 224]))
        zone_counts['outer'] += 1
    if not zoneAA.is_empty:
        output_features.append(create_geojson_feature(zoneAA, "zone_aa", [128, 255, 128]))
        zone_counts['inner'] += 1
    if not zoneBB.is_empty:
        output_features.append(create_geojson_feature(zoneBB, "zone_bb", [160, 255, 160]))
        zone_counts['inner'] += 1
    if not zoneCC.is_empty:
        output_features.append(create_geojson_feature(zoneCC, "zone_cc", [192, 255, 192]))
        zone_counts['inner'] += 1
    if not zoneDD.is_empty:
        output_features.append(create_geojson_feature(zoneDD, "zone_dd", [208, 255, 208]))
        zone_counts['inner'] += 1

    logger.info(f"Created {zone_counts['compartments']} compartments, {zone_counts['outer']} outer zones, {zone_counts['inner']} inner zones")

    # 5. Save output with compact format
    output_data = {"type": "FeatureCollection", "features": output_features}
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_geojson_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    save_compact_geojson(output_data, output_geojson_path)
    
    logger.info(f"Saved {len(output_features)} final features to {os.path.basename(output_geojson_path)}")
    return len(output_features)


def process_all_slides_stage1b(
    refined_tissue_dir=None,
    output_dir=None,
    tissue_classes=["epithelia", "tumor", "stroma"],
    endothelia_classes=["endothelial", "endothelia"],
    buffer_size=40,
    min_area=3000
):
    """
    Process all manually refined tissue files for Stage 1B.
    """
    logger.info("Starting Stage 1B batch processing...")
    
    refined_tissue_dir = refined_tissue_dir or "../outputs/processed_annotations/stage1a_merged_tissue"
    output_dir = output_dir or "../outputs/processed_annotations/stage1b_compartments_zones"

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Find manually refined files (saved from QuPath with same naming as Stage 1A)
    pattern = os.path.join(refined_tissue_dir, "*_merged_tissue.geojson")
    refined_files = glob.glob(pattern)
    
    if not refined_files:
        logger.warning(f"No manually refined files found matching pattern: {pattern}")
        logger.info("Expected workflow:")
        logger.info("1. Stage 1A outputs *_merged_tissue.geojson files")
        logger.info("2. Import these files into QuPath")
        logger.info("3. Edit the 'tissue' polygons")
        logger.info("4. Save with the SAME filename (overwrites)")
        logger.info("5. Stage 1B processes these edited files")
        return results
    
    logger.info(f"Found {len(refined_files)} refined files to process")
    
    for refined_file in refined_files:
        # Extract base name: "P11_roi_1.ome.tiff - Merged Image_merged_tissue.geojson" -> "P11_roi_1"
        refined_basename = os.path.basename(refined_file)
        slide_base = refined_basename.replace("_merged_tissue.geojson", "").replace(".ome.tiff - Merged Image", "")
        
        logger.info(f"\nProcessing slide: {slide_base}")
        logger.info(f"  Input: {os.path.basename(refined_file)}")
        
        # Generate output path
        output_filename = f"{slide_base}_final_annotations.geojson"
        output_file = os.path.join(output_dir, output_filename)
        
        logger.info(f"  Output: {output_filename}")
        
        try:
            start_time = datetime.now()
            result = create_compartments_and_zones(
                refined_tissue_geojson_path=refined_file,
                output_geojson_path=output_file,
                tissue_classes=tissue_classes,
                endothelia_classes=endothelia_classes,
                buffer_size=buffer_size,
                min_area=min_area
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results[slide_base] = {
                'success': True,
                'output_path': output_file,
                'total_features': result,
                'refined_file': refined_file,
                'processing_time': processing_time
            }
            
            logger.info(f"  Success: {result} features created in {processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"  Error processing {slide_base}: {e}")
            logger.error(traceback.format_exc())
            results[slide_base] = {
                'success': False,
                'error': str(e),
                'refined_file': refined_file
            }
    
    successful = sum(1 for r in results.values() if r.get('success'))
    total_time = sum(r.get('processing_time', 0) for r in results.values() if r.get('success'))
    total_features = sum(r.get('total_features', 0) for r in results.values() if r.get('success'))
    
    logger.info(f"\nStage 1B processing complete: {successful}/{len(results)} files successful")
    if successful > 0:
        logger.info(f"Total processing time: {total_time:.1f}s")
        logger.info(f"Average per file: {total_time/successful:.1f}s")
        logger.info(f"Total features created: {total_features}")
    
    return results


def efficient_union(polygons, batch_size=500):
    """
    Efficiently merge a large number of polygons using batched processing.
    
    Args:
        polygons: List of shapely polygons to merge
        batch_size: Number of polygons to process in each batch
    
    Returns:
        Merged polygon or multipolygon
    """
    if not polygons:
        return None
    
    if len(polygons) == 1:
        return polygons[0]
    
    # Process in batches to avoid memory issues
    logger.info(f"Processing {len(polygons)} polygons in batches of {batch_size}")
    
    # Create batches
    batches = [polygons[i:i + batch_size] for i in range(0, len(polygons), batch_size)]
    
    # Process each batch
    batch_results = []
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} polygons")
        try:
            # Fix invalid geometries before union
            batch = [geom.buffer(0) if not geom.is_valid else geom for geom in batch]
            batch_union = unary_union(batch)
            if batch_union and not batch_union.is_empty:
                batch_results.append(batch_union)
        except Exception as e:
            logger.warning(f"Error processing batch {i+1}: {e}")
            continue
    
    # Final merge of batch results
    if not batch_results:
        return None
    
    if len(batch_results) == 1:
        return batch_results[0]
    
    logger.info(f"Final merge of {len(batch_results)} batch results")
    # Fix invalid geometries before final union
    batch_results = [geom.buffer(0) if not geom.is_valid else geom for geom in batch_results]
    return unary_union(batch_results)

def save_compact_geojson(output_data, output_path):
    """
    Save GeoJSON in compact format to reduce file size.
    
    - No indentation (compact format)
    - Round coordinates to 1 decimal place for good precision with smaller file size
    - Remove unnecessary whitespace
    """
    # Round coordinates to reduce precision and file size
    def round_coords(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'Polygon':
                # Round polygon coordinates to 1 decimal place
                coords = obj['coordinates']
                obj['coordinates'] = [[[round(pt[0], 1), round(pt[1], 1)] for pt in ring] for ring in coords]
            elif obj.get('type') == 'MultiPolygon':
                # Round multipolygon coordinates to 1 decimal place
                coords = obj['coordinates']
                obj['coordinates'] = [[[[round(pt[0], 1), round(pt[1], 1)] for pt in ring] for ring in polygon] for polygon in coords]
            else:
                # Recursively process other objects
                for key, value in obj.items():
                    round_coords(value)
        elif isinstance(obj, list):
            for item in obj:
                round_coords(item)
        return obj
    
    # Round coordinates in the output data
    round_coords(output_data)
    
    # Save without indentation for compact format
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directories if there's a directory path
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, separators=(',', ':'))


class ROIInfo:
    """Information about a single ROI region."""
    def __init__(self, roi_id, total_area, source_file):
        self.roi_id = roi_id
        self.total_area = total_area
        self.source_file = source_file


class ROITracker:
    """
    Simple ROI tracking system for managing region information.
    
    This provides a basic interface for tracking ROI information
    without complex dependencies.
    """
    def __init__(self):
        self.rois = {}  # Dictionary of roi_id -> ROIInfo
        self.current_roi_id = 1
    
    def add_roi(self, total_area, source_file):
        """Add a new ROI and return its ID."""
        roi_id = f"ROI_{self.current_roi_id:03d}"
        self.rois[roi_id] = ROIInfo(roi_id, total_area, source_file)
        self.current_roi_id += 1
        return roi_id
    
    def get_roi_info(self, roi_id):
        """Get ROI information by ID."""
        return self.rois.get(roi_id)



def process_all_slides_stage2(stage1b_dir, cells_dir, output_dir, roi_tracker=None, csv_pattern="*_features.csv", overwrite=False):
    """
    Process all slides for Stage 2 annotation.
    
    Args:
        stage1b_dir: Directory containing Stage 1B GeoJSON files
        cells_dir: Directory containing segmentation CSV files  
        output_dir: Directory for output annotated CSV files
        roi_tracker: ROI tracking object (optional)
        csv_pattern: Pattern to match CSV files
        overwrite: Whether to overwrite existing output files
        
    Returns:
        Dictionary with processing results
    """
    return process_all_csvs_stage2(
        csv_dir=cells_dir,
        geojson_dir=stage1b_dir, 
        output_dir=output_dir,
        csv_pattern=csv_pattern,
        overwrite=overwrite
    )
