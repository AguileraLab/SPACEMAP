"""acc_utils.py
Utilities for extracting manually annotated (ground truth) phenotyping data
from GeoJSON files into CSV format.

The notebook should import the public functions:
	- extract_classification_data
	- batch_process_geojson_files

These functions were factored out of the Step7_accessories notebook so the
core logic is centralized and easier to test / reuse. Only lightweight
configuration (paths, patterns) and the actual invocation remain in the
notebook.
"""
from __future__ import annotations
import os
import json
import glob
import re
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

__all__ = [
	"extract_classification_data",
	"batch_process_geojson_files",
	"merge_ground_truth_into_unified",
	"parse_gt_prefix",
	"relabel_ground_truth_files",
	"annotate_geojson_with_phenotypes",
	"batch_annotate_geojson",
	"enrich_ground_truth_with_unified_data",
    "generate_png_from_geojson",
	"build_color_mapping",
	"DEFAULT_PHENOTYPE_COLORS",
	"DEFAULT_MATCH_COLOR",
	"DEFAULT_MISMATCH_COLOR",
	"DEFAULT_IGNORED_COLOR",
	"default_palette",
	"feature_label",
	"find_discrepant_circles",
]


def extract_classification_data(geojson_path: str, output_csv_path: str, *, progress: bool = True) -> bool:
	"""Extract classification data from a phenotyped GeoJSON file and save to CSV.

	Parameters
	----------
	geojson_path : str
		Path to the phenotyped GeoJSON file.
	output_csv_path : str
		Path to save the extracted classification data as CSV.
	progress : bool, default True
		If True, prints progress / summary information.

	Returns
	-------
	bool
		True if successful, False otherwise.
	"""
	if progress:
		print(f"Starting extraction from: {geojson_path}")
		print(f"Output will be saved to: {output_csv_path}")
		print(f"Process started at {datetime.now().strftime('%H:%M:%S')}")

	# 1. Load the GeoJSON file
	try:
		with open(geojson_path, "r") as f:
			geojson_data = json.load(f)
	except Exception as e:  # noqa: BLE001
		if progress:
			print(f"ERROR: Failed to load GeoJSON file: {e}")
		return False

	if "features" not in geojson_data or not isinstance(geojson_data["features"], list):
		if progress:
			print("ERROR: Invalid GeoJSON format - missing or invalid 'features' key")
		return False

	features = geojson_data["features"]
	total_features = len(features)
	if progress:
		print(f"Found {total_features} features in GeoJSON")

	if total_features == 0:
		if progress:
			print("WARNING: No features found in GeoJSON file")
		return False

	# 2. Extract data from each feature
	extracted_data = []
	classified_count = 0
	unclassified_count = 0

	if progress:
		print("Extracting data from features...")
	print_interval = max(1, total_features // 20)  # every ~5%

	for idx, feature in enumerate(features):
		try:
			geom_data = feature.get("geometry")
			if not geom_data or geom_data.get("type") not in ["Polygon", "MultiPolygon"]:
				if progress:
					print(f"WARNING: Feature {idx} has invalid geometry, skipping")
				continue

			polygon = shape(geom_data)

			centroid = polygon.centroid
			x_centroid = centroid.x
			y_centroid = centroid.y
			area = polygon.area

			properties = feature.get("properties", {})
			classification_data = properties.get("classification", {})

			if isinstance(classification_data, dict):
				classification = classification_data.get("name", "unclassified")
			elif isinstance(classification_data, str):
				classification = classification_data
			else:
				classification = "unclassified"

			cell_id = f"cell_{idx + 1:06d}"

			extracted_data.append(
				{
					"cell_id": cell_id,
					"x_centroid": round(x_centroid, 2),
					"y_centroid": round(y_centroid, 2),
					"classification": classification,
					"area_pixels": round(area, 2),
				}
			)

			if classification != "unclassified":
				classified_count += 1
			else:
				unclassified_count += 1

		except Exception as e:  # noqa: BLE001
			if progress:
				print(f"WARNING: Error processing feature {idx}: {e}")
			continue

		if progress and ((idx + 1) % print_interval == 0 or (idx + 1) == total_features):
			percent = ((idx + 1) / total_features) * 100
			print(f"Processed {idx + 1} / {total_features} features ({percent:.1f}%)")

	if not extracted_data:
		if progress:
			print("ERROR: No valid data extracted from GeoJSON")
		return False

	df = pd.DataFrame(extracted_data).sort_values("cell_id").reset_index(drop=True)

	try:
		df.to_csv(output_csv_path, index=False)
		if progress:
			print(f"\nSuccessfully saved {len(df)} cell records to: {output_csv_path}")
	except Exception as e:  # noqa: BLE001
		if progress:
			print(f"ERROR: Failed to save CSV file: {e}")
		return False

	if progress:
		print("\n" + "=" * 60)
		print("EXTRACTION SUMMARY")
		print("=" * 60)
		print(f"Total cells processed: {len(df)}")
		print(f"Classified cells: {classified_count}")
		print(f"Unclassified cells: {unclassified_count}")

		if classified_count > 0:
			print("\nClassification distribution:")
			class_counts = df[df["classification"] != "unclassified"]["classification"].value_counts()
			for phenotype, count in class_counts.items():
				percentage = (count / len(df)) * 100
				print(f"  {phenotype}: {count} cells ({percentage:.1f}%)")

		print("\nArea statistics (square pixels):")
		print(f"  Mean area: {df['area_pixels'].mean():.2f}")
		print(f"  Median area: {df['area_pixels'].median():.2f}")
		print(f"  Min area: {df['area_pixels'].min():.2f}")
		print(f"  Max area: {df['area_pixels'].max():.2f}")
		print(f"\nExtraction completed at {datetime.now().strftime('%H:%M:%S')}")

	return True


def batch_process_geojson_files(input_directory: str, output_directory: str, file_pattern: str = "*.geojson") -> None:
	"""Process multiple GeoJSON files in a directory.

	Parameters
	----------
	input_directory : str
		Directory containing phenotyped GeoJSON files.
	output_directory : str
		Directory to save the extracted CSV files.
	file_pattern : str, default '*.geojson'
		Glob pattern for GeoJSON files.
	"""
	search_pattern = os.path.join(input_directory, file_pattern)
	geojson_files: List[str] = glob.glob(search_pattern)

	if not geojson_files:
		print(f"No GeoJSON files found matching pattern: {search_pattern}")
		return

	print(f"Found {len(geojson_files)} GeoJSON files to process:")
	for f in geojson_files:
		print(f"  - {os.path.basename(f)}")

	os.makedirs(output_directory, exist_ok=True)

	successful = 0
	failed = 0

	for geojson_file in geojson_files:
		print(f"\n{'=' * 80}")
		print(f"Processing: {os.path.basename(geojson_file)}")
		print(f"{'=' * 80}")
		base_name = os.path.splitext(os.path.basename(geojson_file))[0]
		csv_name = f"{base_name}_ground_truth.csv"
		output_csv = os.path.join(output_directory, csv_name)
		if extract_classification_data(geojson_file, output_csv):
			successful += 1
		else:
			failed += 1

	print(f"\n{'=' * 80}")
	print("BATCH PROCESSING COMPLETE")
	print(f"{'=' * 80}")
	print(f"Successfully processed: {successful} files")
	print(f"Failed to process: {failed} files")


# -----------------------------------------------------------------------------
# Ground truth merging into unified phenotyping file
# -----------------------------------------------------------------------------
import re
from collections import defaultdict


def parse_gt_prefix(filename: str, custom_pattern: str | None = None) -> str:
	"""Extract dataset prefix from a ground truth filename using flexible patterns.

	Parameters
	----------
	filename : str
		The ground truth CSV filename.
	custom_pattern : str, optional
		Custom regex pattern to extract prefix. If None, tries multiple common patterns.

	Examples
	--------
	P11_roi_1.ome.tiff - Merged Image_ground_truth.csv -> P11_roi_1
	reg018_B_ground_truth.csv -> reg018_B
	P17_P19_roi_1_7_circle1_ground_truth.csv -> P17_P19_roi_1_7_circle1
	
	Returns empty string if no match.
	"""
	if custom_pattern:
		m = re.search(custom_pattern, filename)
		return m.group(1) if m else ""
	
	# Try multiple common patterns
	patterns = [
		r"^(.*?)\.ome\.tiff",  # Original pattern: P11_roi_1.ome.tiff - ...
		r"^(.+)_ground_truth\.csv$",  # Simple pattern: reg018_B_ground_truth.csv -> reg018_B
		r"^(.+)_gt\.csv$",  # Alternative: reg018_B_gt.csv -> reg018_B
		r"^(.+)\.csv$",  # Fallback: just remove .csv extension
	]
	
	for pattern in patterns:
		m = re.search(pattern, filename)
		if m:
			return m.group(1)
	
	return ""


def merge_ground_truth_into_unified(
	gt_output_dir: str,
	unified_csv_path: str,
	*,
	output_path: str | None = None,
	phenotype_col_name: str = "phenotype_ref1",
	file_pattern: str = "*_ground_truth.csv",
	tolerance_fraction: float = 0.01,
	override_existing: bool = True,
	exclude_classifications: list[str] | None = None,
	batch_id_column: str = "batch_id",
	x_centroid_column: str = "x_centroid",
	y_centroid_column: str = "y_centroid",
	gt_x_centroid_column: str = "x_centroid",
	gt_y_centroid_column: str = "y_centroid",
	gt_classification_column: str = "classification",
	custom_prefix_pattern: str | None = None,
) -> pd.DataFrame:
	"""Merge ground truth classifications from CSV files into a unified phenotyping file.

	Parameters
	----------
	gt_output_dir : str
		Directory containing the ground truth CSV files.
	unified_csv_path : str
		Path to the unified phenotyping CSV file.
	output_path : str, optional
		Path to save the updated unified file. If None, overwrites `unified_csv_path`.
	phenotype_col_name : str, default 'phenotype_ref1'
		Name of the column to add/update with ground truth classifications.
	file_pattern : str, default '*_ground_truth.csv'
		Pattern to match ground truth CSV files.
	tolerance_fraction : float, default 0.01
		Positional tolerance for matching cells, as a fraction of the batch's max coordinate span.
	override_existing : bool, default True
		If True, overrides existing values in the phenotype column.
	exclude_classifications : list of str, optional
		A list of classification labels to exclude from the merge.

	Returns
	-------
	pd.DataFrame
		The updated unified dataframe.
	"""
	if not os.path.exists(unified_csv_path):
		raise FileNotFoundError(f"Unified phenotyping file not found: {unified_csv_path}")

	gt_files = glob.glob(os.path.join(gt_output_dir, file_pattern))
	if not gt_files:
		raise FileNotFoundError(f"No ground truth files found in {gt_output_dir} matching {file_pattern}")

	print(f"Found {len(gt_files)} ground truth CSV files to merge.")

	unified_df = pd.read_csv(unified_csv_path)
	if batch_id_column not in unified_df.columns:
		raise KeyError(f"'{batch_id_column}' column missing from unified phenotyping file.")
	if x_centroid_column not in unified_df.columns or y_centroid_column not in unified_df.columns:
		raise KeyError(f"Unified file must contain '{x_centroid_column}' and '{y_centroid_column}' columns for matching.")

	# Prepare target column
	if phenotype_col_name not in unified_df.columns:
		unified_df[phenotype_col_name] = pd.NA

	total_gt_cells = 0
	total_matched_gt = 0
	per_batch_stats = []
	conflict_examples = []

	for gt_file in gt_files:
		base_name = os.path.basename(gt_file)
		prefix = parse_gt_prefix(base_name, custom_prefix_pattern)
		if not prefix:
			print(f"Warning: Could not parse prefix from {base_name}; skipping.")
			print(f"  Tried patterns: {custom_prefix_pattern if custom_prefix_pattern else 'default patterns'}")
			print(f"  Consider setting a custom_prefix_pattern parameter for your filename format.")
			continue

		try:
			gt_df = pd.read_csv(gt_file)
		except Exception as e:  # noqa: BLE001
			print(f"  ERROR loading {base_name}: {e}")
			continue

		required_cols = {gt_x_centroid_column, gt_y_centroid_column, gt_classification_column}
		if not required_cols.issubset(gt_df.columns):
			print(f"  WARNING: Missing required columns in {base_name}; has {gt_df.columns}")
			continue

		# Filter out excluded classifications
		if exclude_classifications:
			initial_count = len(gt_df)
			gt_df = gt_df[~gt_df[gt_classification_column].isin(exclude_classifications)]
			if len(gt_df) < initial_count:
				print(f"  Filtered out {initial_count - len(gt_df)} rows from {base_name} based on exclude list.")

		if gt_df.empty:
			print(f"  INFO: No classified cells remain in {base_name} after filtering; skipping.")
			continue

		batch_mask = unified_df[batch_id_column].astype(str) == prefix
		batch_df = unified_df.loc[batch_mask]
		if batch_df.empty:
			print(f"  WARNING: No unified rows with {batch_id_column} matching '{prefix}'")
			continue

		# Compute per-batch tolerance in pixels
		x_range = batch_df[x_centroid_column].max() - batch_df[x_centroid_column].min()
		y_range = batch_df[y_centroid_column].max() - batch_df[y_centroid_column].min()
		max_range = max(x_range, y_range, 1.0)  # avoid zero
		tolerance_pixels = tolerance_fraction * max_range
		print(f"  Processing {base_name} | prefix={prefix} | GT cells={len(gt_df)} | tolerance={tolerance_pixels:.3f} px")

		# Build arrays for distance calc
		batch_coords = batch_df[[x_centroid_column, y_centroid_column]].to_numpy()
		gt_coords = gt_df[[gt_x_centroid_column, gt_y_centroid_column]].to_numpy()

		# Map from unified index -> (distance, classification)
		chosen_matches: dict[int, tuple[float, str]] = {}
		matched_gt_count = 0

		for i, (gx, gy) in enumerate(gt_coords):
			dists = np.sqrt(((batch_coords - np.array([gx, gy])) ** 2).sum(axis=1))
			min_idx = dists.argmin()
			dist = dists[min_idx]
			if dist <= tolerance_pixels:
				unified_index = batch_df.index[min_idx]
				classification = str(gt_df.iloc[i][gt_classification_column])  # ensure string
				if unified_index in chosen_matches:
					prev_dist, prev_class = chosen_matches[unified_index]
					if dist < prev_dist:
						if prev_class != classification:
							conflict_examples.append((prefix, unified_index, prev_class, classification))
						chosen_matches[unified_index] = (dist, classification)
				else:
					chosen_matches[unified_index] = (dist, classification)
				matched_gt_count += 1

		# Apply matches
		applied_matches = 0
		for unified_index, (_, classification) in chosen_matches.items():
			if (not override_existing) and pd.notna(unified_df.at[unified_index, phenotype_col_name]):
				continue
			unified_df.at[unified_index, phenotype_col_name] = classification
			applied_matches += 1

		total_gt_cells += len(gt_df)
		total_matched_gt += applied_matches

		match_pct = (applied_matches / len(gt_df) * 100) if len(gt_df) else 0.0
		per_batch_stats.append(
			{
				"prefix": prefix,
				"gt_cells": len(gt_df),
				"matched_gt_cells": applied_matches,
				"match_pct": match_pct,
				"tolerance_pixels": tolerance_pixels,
			}
		)
		print(f"    Matched {applied_matches}/{len(gt_df)} GT cells ({match_pct:.1f}%) for prefix {prefix}")

	if not per_batch_stats:
		print("No ground truth classifications merged.")
		return unified_df

	overall_pct = (total_matched_gt / total_gt_cells * 100) if total_gt_cells else 0.0
	print("\n=== Ground Truth Merge Summary ===")
	print(f"Total GT classified cells: {total_gt_cells}")
	print(f"Total matched into unified: {total_matched_gt} ({overall_pct:.1f}%)")
	print(f"Phenotype column: {phenotype_col_name}")
	if conflict_examples:
		print(f"Conflicts encountered (showing up to 5): {len(conflict_examples)}")
		for c in conflict_examples[:5]:
			print(f"  Prefix {c[0]} unified_index {c[1]} prev={c[2]} new={c[3]}")

	stats_df = pd.DataFrame(per_batch_stats)

	# Save updated unified file
	if output_path is None:
		output_path = unified_csv_path
	unified_df.to_csv(output_path, index=False)
	print(f"Updated unified phenotyping file saved: {output_path}")

	# Save stats
	stats_out = os.path.join(os.path.dirname(output_path), f"{phenotype_col_name}_merge_stats.csv")
	stats_df.to_csv(stats_out, index=False)
	print(f"Per-batch merge stats saved: {stats_out}")

	return unified_df


def enrich_ground_truth_with_unified_data(
	gt_output_dir: str,
	unified_csv_path: str,
	file_pattern: str = "*_ground_truth.csv",
	tolerance_fraction: float = 0.01,
	exclude_columns: list[str] | None = None,
	batch_id_column: str = "batch_id",
	x_centroid_column: str = "x_centroid",
	y_centroid_column: str = "y_centroid",
	gt_x_centroid_column: str = "x_centroid",
	gt_y_centroid_column: str = "y_centroid",
) -> dict[str, int]:
	"""Enrich ground truth CSV files with additional columns from the unified phenotyping file.

	This function matches cells between ground truth files and the unified phenotyping file
	based on coordinates, then adds marker intensities and other features to the ground truth files.

	Parameters
	----------
	gt_output_dir : str
		Directory containing ground truth CSV files.
	unified_csv_path : str
		Path to the unified phenotyping CSV file.
	file_pattern : str, default '*_ground_truth.csv'
		Glob pattern for ground truth CSV files.
	tolerance_fraction : float, default 0.01
		Fraction (1% default) of the max(x_range, y_range) per batch used as
		positional tolerance when matching cells.
	exclude_columns : list[str], optional
		List of column names to exclude from enrichment (e.g., existing phenotype columns).
		If None, excludes common conflicting columns.

	Returns
	-------
	dict[str, int]
		Dictionary with file names as keys and number of enriched cells as values.
	"""
	if not os.path.exists(unified_csv_path):
		raise FileNotFoundError(f"Unified phenotyping file not found: {unified_csv_path}")

	gt_files = glob.glob(os.path.join(gt_output_dir, file_pattern))
	if not gt_files:
		print(f"No ground truth files found in {gt_output_dir} matching {file_pattern}")
		return {}

	print(f"Loading unified phenotyping file: {unified_csv_path}")
	unified_df = pd.read_csv(unified_csv_path)
	
	# Validate required columns
	required_cols = [batch_id_column, x_centroid_column, y_centroid_column]
	if not all(col in unified_df.columns for col in required_cols):
		missing = [col for col in required_cols if col not in unified_df.columns]
		raise KeyError(f"Unified file missing required columns: {missing}")

	# Set default exclude columns if not provided
	if exclude_columns is None:
		exclude_columns = [
			"cell_id", gt_x_centroid_column, gt_y_centroid_column, "classification",  # Already in GT files
			batch_id_column, "tissue_roi",  # Structural columns that might cause conflicts
		]

	# Get columns to add (all unified columns except excluded ones)
	columns_to_add = [col for col in unified_df.columns if col not in exclude_columns]
	
	print(f"Will enrich ground truth files with {len(columns_to_add)} additional columns:")
	print(f"  {', '.join(columns_to_add[:10])}" + ("..." if len(columns_to_add) > 10 else ""))

	enrichment_stats = {}

	for gt_path in gt_files:
		base_name = os.path.basename(gt_path)
		prefix = parse_gt_prefix(base_name)
		if not prefix:
			print(f"Warning: Could not parse prefix from {base_name}; skipping.")
			continue

		try:
			gt_df = pd.read_csv(gt_path)
		except Exception as e:
			print(f"  ERROR loading {base_name}: {e}")
			continue

		# Validate ground truth file has required columns
		required_gt_cols = {gt_x_centroid_column, gt_y_centroid_column, "classification"}
		if not required_gt_cols.issubset(gt_df.columns):
			print(f"  WARNING: {base_name} missing required columns; skipping")
			continue

		# Find matching batch data in unified file
		batch_mask = unified_df[batch_id_column].astype(str).str.startswith(prefix)
		batch_df = unified_df.loc[batch_mask]
		if batch_df.empty:
			print(f"  WARNING: No unified rows with {batch_id_column} starting '{prefix}' for {base_name}")
			enrichment_stats[base_name] = 0
			continue

		# Compute tolerance for this batch
		x_range = batch_df[x_centroid_column].max() - batch_df[x_centroid_column].min()
		y_range = batch_df[y_centroid_column].max() - batch_df[y_centroid_column].min()
		max_range = max(x_range, y_range, 1.0)
		tolerance_pixels = tolerance_fraction * max_range

		print(f"  Processing {base_name} | prefix={prefix} | GT cells={len(gt_df)} | tolerance={tolerance_pixels:.3f} px")

		# Match cells and collect additional data
		batch_coords = batch_df[[x_centroid_column, y_centroid_column]].to_numpy()
		gt_coords = gt_df[[gt_x_centroid_column, gt_y_centroid_column]].to_numpy()

		# For each GT cell, find the closest unified cell within tolerance
		enriched_rows = []
		matched_count = 0

		for i, (gx, gy) in enumerate(gt_coords):
			# Calculate distances to all batch cells
			dists = np.sqrt(((batch_coords - np.array([gx, gy])) ** 2).sum(axis=1))
			min_idx = dists.argmin()
			min_dist = dists[min_idx]

			# Start with the original GT row
			enriched_row = gt_df.iloc[i].to_dict()

			if min_dist <= tolerance_pixels:
				# Add data from the matched unified cell
				unified_row = batch_df.iloc[min_idx]
				for col in columns_to_add:
					enriched_row[col] = unified_row[col]
				matched_count += 1
			else:
				# No match found, add NaN for the additional columns
				for col in columns_to_add:
					enriched_row[col] = pd.NA

			enriched_rows.append(enriched_row)

		# Create enriched DataFrame and save
		enriched_df = pd.DataFrame(enriched_rows)
		
		# Save the enriched file (overwrite original)
		try:
			enriched_df.to_csv(gt_path, index=False)
			match_pct = (matched_count / len(gt_df) * 100) if len(gt_df) else 0.0
			print(f"    Enriched {matched_count}/{len(gt_df)} cells ({match_pct:.1f}%) with unified data")
			enrichment_stats[base_name] = matched_count
		except Exception as e:
			print(f"    ERROR saving enriched file {gt_path}: {e}")
			enrichment_stats[base_name] = 0

	# Summary
	total_files = len(enrichment_stats)
	successful_files = sum(1 for count in enrichment_stats.values() if count > 0)
	total_enriched = sum(enrichment_stats.values())
	
	print(f"\n=== Ground Truth Enrichment Summary ===")
	print(f"Processed {total_files} files | Successfully enriched {successful_files} files")
	print(f"Total cells enriched with unified data: {total_enriched}")
	print(f"Added columns: {', '.join(columns_to_add)}")

	return enrichment_stats


def relabel_ground_truth_files(
	gt_output_dir: str,
	mapping: dict[str, str],
	file_pattern: str = "*_ground_truth.csv",
	case_insensitive: bool = True,
	dry_run: bool = False,
) -> pd.DataFrame:
	"""Relabel / merge classification categories inside ground truth CSV files.

	Useful to merge rarer or synonymous classes (e.g. MDSC -> Granulocyte) prior
	to merging into the unified phenotyping file. Operates in-place unless
	dry_run=True.

	Parameters
	----------
	gt_output_dir : str
		Directory containing ground truth CSVs (post extraction).
	mapping : dict[str, str]
		Dict mapping source_class -> target_class. Multiple sources can point
		to the same target for merging.
	file_pattern : str, default '*_ground_truth.csv'
		Glob pattern of files to process.
	case_insensitive : bool, default True
		If True, matching is performed case-insensitively.
	dry_run : bool, default False
		If True, no files are written; only a summary is returned.

	Returns
	-------
	pd.DataFrame
		Summary table with per-file relabel statistics.
	"""
	if not mapping:
		print("No mapping provided; skipping relabel.")
		return pd.DataFrame()

	files = glob.glob(os.path.join(gt_output_dir, file_pattern))
	if not files:
		print(f"No ground truth files found to relabel in {gt_output_dir} matching {file_pattern}")
		return pd.DataFrame()

	# Prepare mapping (optionally case-insensitive)
	if case_insensitive:
		norm_map = {k.lower(): v for k, v in mapping.items()}
	else:
		norm_map = mapping.copy()

	records: list[dict] = []
	for path in files:
		try:
			df = pd.read_csv(path)
		except Exception as e:  # noqa: BLE001
			print(f"  ERROR reading {path}: {e}")
			continue
		if "classification" not in df.columns:
			print(f"  WARNING: 'classification' column missing in {os.path.basename(path)}; skipping")
			continue

		original_classes = df["classification"].astype(str).tolist()
		original_unique = set(original_classes)
		changes = 0
		new_vals = []
		for val in original_classes:
			key = val.lower() if case_insensitive else val
			if key in norm_map:
				target = norm_map[key]
				if target != val:
					changes += 1
				new_vals.append(target)
			else:
				new_vals.append(val)
		if changes:
			print(f"  Relabeled {changes} entries in {os.path.basename(path)}")
		else:
			print(f"  No relabel changes for {os.path.basename(path)}")

		if not dry_run and changes:
			df["classification"] = new_vals
			try:
				df.to_csv(path, index=False)
			except Exception as e:  # noqa: BLE001
				print(f"    ERROR writing relabeled file {path}: {e}")

		new_unique = set(df["classification"].astype(str)) if changes else original_unique
		records.append(
			{
				"file": os.path.basename(path),
				"original_unique_classes": len(original_unique),
				"new_unique_classes": len(new_unique),
				"changes_applied": changes if not dry_run else 0,
				"dry_run": dry_run,
			}
		)

	summary_df = pd.DataFrame(records)
	if not summary_df.empty:
		out_summary = os.path.join(gt_output_dir, "relabel_summary.csv")
		if not dry_run:
			summary_df.to_csv(out_summary, index=False)
			print(f"Relabel summary saved: {out_summary}")
	return summary_df


DEFAULT_UNASSIGNED_LABEL = "unassigned"
DEFAULT_UNASSIGNED_COLOR = "#D9D9D9"

# The house phenotype palette, matching the figure scripts in test_codes/ so a
# GeoJSON overlay and a published figure use one color per cell type. Labels are
# matched case-insensitively; anything not listed falls through to the cycling
# palette below. Colors edited in the GUI are stored per project and win over
# these.
# Values match PHENOTYPE_COLORS in test_codes/Fig3_code_results/fig3_code_v1.py,
# so the GUI overlays, the GeoJSON exports and the published Figure 3 all render
# a given cell type in the same colour.
DEFAULT_PHENOTYPE_COLORS: dict[str, str] = {
    "B": "#24306e",
    "CD4": "#e78ac3",
    "CD8": "#e31a1c",
    "DC": "#2ca02c",
    "Endothelial": "#006d6f",
    "Epithelial": "#7b3294",
    "Fibroblast": "#737373",
    "IMC": "#348abd",
    "Macrophage": "#9ecae1",
    "Plasma": "#b65cff",
    "Treg": "#ff9800",
    "Others": "#d9d9d9",
}

# Aliases for the same cell types under the names other methods emit.
DEFAULT_PHENOTYPE_COLORS.update(
    {
        "B cell": DEFAULT_PHENOTYPE_COLORS["B"],
        "Plasma B cell": DEFAULT_PHENOTYPE_COLORS["Plasma"],
        "CAF": DEFAULT_PHENOTYPE_COLORS["Fibroblast"],
        "Endothelia": DEFAULT_PHENOTYPE_COLORS["Endothelial"],
        "Epithelia": DEFAULT_PHENOTYPE_COLORS["Epithelial"],
        "Mac": DEFAULT_PHENOTYPE_COLORS["Macrophage"],
        "Other": DEFAULT_PHENOTYPE_COLORS["Others"],
    }
)

# Agreement-map defaults, applied by role rather than by name so they still hold
# when the labels are renamed (for example "Mismatch" -> "Missmatch").
DEFAULT_MATCH_COLOR = "#00693C"
DEFAULT_MISMATCH_COLOR = "#D62728"
DEFAULT_IGNORED_COLOR = "#D9D9D9"

# tab20 + tab20b + tab20c gives 60 visually distinct colors before any repeat,
# which covers every phenotype panel this pipeline produces in practice.
_PALETTE_CMAPS = ("tab20", "tab20b", "tab20c")


def _hex_from_rgb(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)), int(round(rgb[1] * 255)), int(round(rgb[2] * 255))
    )


def default_palette(n: int) -> list[str]:
    """Return `n` distinct hex colors, cycling once the palettes are exhausted."""
    colors: list[str] = []
    for cmap_name in _PALETTE_CMAPS:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(_hex_from_rgb(cmap(i)) for i in range(cmap.N))
    if n <= 0:
        return []
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def _srgb_to_lab(hex_color: str) -> tuple[float, float, float]:
    """Convert a hex colour to CIE Lab (D65), for perceptual distance."""
    r, g, b = to_rgb(hex_color)

    def _linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = _linear(r), _linear(g), _linear(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    return (116.0 * fy) - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _lab_distance(first: str, second: str) -> float:
    a = _srgb_to_lab(first)
    b = _srgb_to_lab(second)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def distinct_colors(n: int, avoid: "list[str] | None" = None, min_distance: float = 25.0) -> list[str]:
    """`n` colours chosen to look unlike each other and unlike `avoid`.

    A plain cycling palette can hand a new phenotype a colour that is visually
    identical to one already in use -- tab20 offers both #9ecae1 and #aec7e8,
    which no reader can tell apart in a dense figure. Candidates are therefore
    scored by CIE Lab distance to everything already assigned and the farthest
    is taken each time (greedy farthest-point). Selection is deterministic, so
    the same panel always yields the same colours.
    """
    if n <= 0:
        return []
    taken = [c for c in (avoid or []) if c]
    candidates = [c for c in default_palette(60) if c not in taken]
    chosen: list[str] = []
    for _ in range(n):
        if not candidates:
            candidates = default_palette(60)
        best = max(
            candidates,
            key=lambda candidate: min(
                (_lab_distance(candidate, other) for other in (taken + chosen)), default=1e9
            ),
        )
        chosen.append(best)
        candidates.remove(best)
    # Report, rather than silently accept, a palette too crowded to separate.
    for colour in chosen:
        closest = min((_lab_distance(colour, o) for o in taken if o != colour), default=1e9)
        if closest < min_distance:
            break
    return chosen


def build_color_mapping(
    labels,
    existing: dict[str, str] | None = None,
    unassigned_label: str = DEFAULT_UNASSIGNED_LABEL,
    unassigned_color: str = DEFAULT_UNASSIGNED_COLOR,
    use_known_colors: bool = True,
) -> dict[str, str]:
    """Assign a hex color to every label, keeping any colors already chosen.

    Resolution order is `existing` (a project's saved choices), then the house
    palette in `DEFAULT_PHENOTYPE_COLORS`, then the cycling fallback. Fallback
    labels are colored in sorted order so the same panel always gets the same
    palette regardless of the order the labels were discovered in.
    """
    mapping = dict(existing or {})
    pending = sorted({str(label) for label in labels if str(label) not in mapping})

    if unassigned_label in pending:
        pending.remove(unassigned_label)
        mapping.setdefault(unassigned_label, unassigned_color)

    if use_known_colors:
        known = {name.strip().lower(): color for name, color in DEFAULT_PHENOTYPE_COLORS.items()}
        still_pending = []
        for label in pending:
            color = known.get(label.strip().lower())
            if color:
                mapping[label] = color
            else:
                still_pending.append(label)
        pending = still_pending

    # Colour the remaining labels away from everything already assigned, so a
    # phenotype outside the house scheme never lands on a near-duplicate of one
    # inside it.
    palette = distinct_colors(len(pending), avoid=list(mapping.values()))
    for label, color in zip(pending, palette):
        mapping[label] = color
    return mapping


def _resolve_color(color_mapping: dict[str, str], label: str, fallback: str = "#000000") -> str:
    color = color_mapping.get(str(label))
    if not color:
        return fallback
    try:
        to_rgb(color)
    except ValueError:
        return fallback
    return color


def _match_polygons_to_points(polygons: list, xs, ys):
    """Map each polygon to the index of a CSV point it contains, or -1.

    Returns a numpy array of length `len(polygons)`. When several points fall
    inside one polygon the lowest CSV row index wins, which keeps the result
    stable across runs.
    """
    assigned = np.full(len(polygons), -1, dtype=np.int64)
    if len(polygons) == 0 or len(xs) == 0:
        return assigned

    points = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    tree = STRtree(points)

    try:
        # Shapely 2.x resolves the whole batch inside GEOS; a per-polygon
        # Python loop over a 500k-cell section takes minutes instead.
        result = tree.query(np.asarray(polygons, dtype=object), predicate="contains")
        poly_idx = np.asarray(result[0], dtype=np.int64)
        point_idx = np.asarray(result[1], dtype=np.int64)
        if len(poly_idx):
            order = np.lexsort((point_idx, poly_idx))
            poly_sorted = poly_idx[order]
            point_sorted = point_idx[order]
            first = np.ones(len(poly_sorted), dtype=bool)
            first[1:] = poly_sorted[1:] != poly_sorted[:-1]
            assigned[poly_sorted[first]] = point_sorted[first]
        return assigned
    except Exception:  # noqa: BLE001 - shapely 1.x has no batched query
        pass

    for p_idx, polygon in enumerate(polygons):
        try:
            candidates = tree.query(polygon)
        except Exception:  # noqa: BLE001
            continue
        best = -1
        for cand in np.atleast_1d(candidates):
            c_idx = int(cand) if not hasattr(cand, "wkb") else points.index(cand)
            if polygon.contains(points[c_idx]) and (best < 0 or c_idx < best):
                best = c_idx
        assigned[p_idx] = best
    return assigned


DEFAULT_NA_TOKENS = ("", "nan", "none", "na", "n/a", "null", "<na>", "unassigned")


def missing_label_mask(series: "pd.Series", na_values=None) -> "pd.Series":
    """Boolean mask of entries that carry no usable label.

    Real NaN/None always count as missing. Text tokens are compared
    case-insensitively after stripping, because a column read through pyarrow
    turns SQL-style nulls into the literal string "None", which would otherwise
    be tagged onto cells as if it were a phenotype.
    """
    tokens = {str(token).strip().lower() for token in (DEFAULT_NA_TOKENS if na_values is None else na_values)}
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isna() | text.isin(tokens)


def build_agreement_labels(
    reference: "pd.Series",
    comparison: "pd.Series",
    *,
    match_label: str = "Match",
    mismatch_label: str = "Mismatch",
    passthrough_values=(),
    na_values=None,
    case_sensitive: bool = False,
) -> "pd.Series":
    """Label each cell by whether `comparison` reproduced the `reference` call.

    Rows whose reference is missing come back as NA so the caller can leave those
    cells untagged. Reference values listed in `passthrough_values` keep their own
    name instead of being scored -- that is how categories such as "Others", which
    were never a confident call to begin with, stay out of the match/mismatch
    tally regardless of what the comparison column says.
    """
    reference_missing = missing_label_mask(reference, na_values)
    comparison_missing = missing_label_mask(comparison, na_values)

    ref_text = reference.astype("string").fillna("")
    cmp_text = comparison.astype("string").fillna("")
    if case_sensitive:
        ref_key, cmp_key = ref_text.str.strip(), cmp_text.str.strip()
    else:
        ref_key = ref_text.str.strip().str.lower()
        cmp_key = cmp_text.str.strip().str.lower()

    passthrough_keys = {
        str(value).strip() if case_sensitive else str(value).strip().lower()
        for value in passthrough_values
    }

    labels = pd.Series(mismatch_label, index=reference.index, dtype=object)
    labels[(ref_key == cmp_key) & ~comparison_missing] = match_label
    if passthrough_keys:
        is_passthrough = ref_key.isin(passthrough_keys)
        labels[is_passthrough] = ref_text[is_passthrough].str.strip()
    labels[reference_missing] = pd.NA
    return labels


def annotate_geojson_with_phenotypes(
    csv_path: str,
    geojson_path: str,
    output_path: str,
    phenotype_col: str,
    filter_criteria: dict[str, list] | None = None,
    color_mapping: dict[str, str] | None = None,
    png_output_path: str | None = None,
    mirror_y_axis: bool = True,
    generate_png: bool = True,
    dpi: int = 150,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    keep_unassigned: bool = False,
    unassigned_label: str = DEFAULT_UNASSIGNED_LABEL,
    unassigned_color: str = DEFAULT_UNASSIGNED_COLOR,
    dataframe: "pd.DataFrame | None" = None,
    stats: dict | None = None,
    png_title: str | None = None,
    na_values=None,
    png_edge_color: str = "white",
    png_line_width: float = 0.5,
    png_alpha: float = 0.8,
    png_background: str = "white",
    compact_output: bool = True,
) -> bool:
    """Read CSV with phenotype data and annotate GeoJSON polygons.

    Every polygon is matched against the cell centroids from the CSV using a
    spatial index; the phenotype of the centroid falling inside the polygon
    becomes the polygon's QuPath `classification`.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with phenotype and centroid data. Ignored when
        `dataframe` is supplied.
    geojson_path : str
        Path to the GeoJSON file with cell boundary polygons.
    output_path : str
        Path to save the annotated GeoJSON file.
    phenotype_col : str
        Name of the column in the CSV containing the phenotype labels.
    filter_criteria : dict, optional
        A dictionary to filter the CSV data. Keys are column names, and values are
        lists of allowed values. e.g., `{'phenotype_leiden': ['CAF', 'T cell']}`.
    color_mapping : dict, optional
        Mapping of phenotype labels to hex color codes. Labels without an entry
        get a color from the default palette.
    x_col, y_col : str
        Centroid column names in the CSV.
    keep_unassigned : bool
        When True, polygons with no matching cell are kept and tagged with
        `unassigned_label`. When False (default) they are dropped from the output.
    na_values : iterable of str, optional
        Text values treated as "no label" on top of real NaN/None. Cells whose
        label is missing take no part in the matching, so they end up unassigned
        rather than being tagged with a literal "None". Defaults to
        `DEFAULT_NA_TOKENS`.
    compact_output : bool
        Write the GeoJSON without indentation, matching the layout the
        segmentation step produces. Pretty-printing inflates a 38 MB section to
        roughly 200 MB.
    dataframe : pandas.DataFrame, optional
        Pre-loaded phenotype table. Avoids re-reading a large CSV per section.
    stats : dict, optional
        Mutable dict filled in with per-file counts (`total_polygons`,
        `matched_polygons`, `unassigned_polygons`, `written_polygons`,
        `csv_rows`, `label_counts`).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    print(f"Starting annotation: {os.path.basename(str(csv_path))} -> {os.path.basename(geojson_path)}")
    color_mapping = dict(color_mapping or {})

    def hex_to_int_color(hex_color: str) -> int:
        return int(hex_color.lstrip("#"), 16)

    # 1. Load the phenotype table and apply filters
    if dataframe is not None:
        df = dataframe.copy()
    else:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"ERROR: CSV file not found: {csv_path}")
            return False

    required_cols = [x_col, y_col, phenotype_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: CSV missing required columns: {missing}")
        return False

    if filter_criteria:
        print(f"  Applying filter criteria: {filter_criteria}")
        mask = pd.Series(True, index=df.index)
        for col, values in filter_criteria.items():
            if col in df.columns:
                mask &= df[col].astype(str).isin([str(v) for v in values])
            else:
                print(f"  Warning: Filter column '{col}' not found in CSV.")
        df = df[mask].copy()
        if df.empty:
            print("  Warning: No rows remaining after filtering. No annotations will be added.")

    df = df.dropna(subset=[x_col, y_col])

    # Cells with no label cannot tag anything. Dropping them here (rather than
    # letting them through as the string "None") is what makes "remove
    # non-assigned cells" behave as its name promises on a sparse column such as
    # a manual reference.
    rows_before_na = len(df)
    df = df[~missing_label_mask(df[phenotype_col], na_values)]
    missing_labels = rows_before_na - len(df)
    if missing_labels:
        print(f"  Skipped {missing_labels} cell(s) with no value in '{phenotype_col}'.")
    print(f"  Found {len(df)} labeled cells to process in CSV after filtering.")

    xs = df[x_col].to_numpy(dtype=float, copy=False)
    ys = df[y_col].to_numpy(dtype=float, copy=False)
    labels = df[phenotype_col].astype(str).to_numpy()
    if len(xs) == 0 and not keep_unassigned:
        print("ERROR: No valid cell centroids found in CSV.")
        return False

    # 2. Read GeoJSON
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: GeoJSON file not found: {geojson_path}")
        return False

    if "features" not in geojson_data or not isinstance(geojson_data["features"], list):
        print("ERROR: Invalid GeoJSON format")
        return False

    # 3. Match polygons to cell centroids
    features: list = []
    polygons: list = []
    for feature in geojson_data["features"]:
        geom = feature.get("geometry")
        if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
            continue
        try:
            polygons.append(shape(geom))
        except Exception:  # noqa: BLE001 - skip malformed geometries
            continue
        features.append(feature)

    total_features = len(features)
    print(f"  Matching {total_features} polygons to {len(xs)} CSV cells...")
    assigned_idx = _match_polygons_to_points(polygons, xs, ys)

    # 4. Complete the color mapping so every written label renders distinctly
    present_labels = {labels[i] for i in assigned_idx if i >= 0}
    if keep_unassigned and (assigned_idx < 0).any():
        present_labels.add(unassigned_label)
    color_mapping = build_color_mapping(
        present_labels,
        existing=color_mapping,
        unassigned_label=unassigned_label,
        unassigned_color=unassigned_color,
    )

    # 5. Tag features
    kept_features = []
    label_counts: dict[str, int] = {}
    matched = 0
    for feature, point_idx in zip(features, assigned_idx):
        if point_idx >= 0:
            label = str(labels[point_idx])
            matched += 1
        elif keep_unassigned:
            label = unassigned_label
        else:
            continue

        # Only `classification` is added. Injecting an `objectType` would move the
        # cells into QuPath's detection hierarchy instead of leaving them where
        # the source file put them.
        properties = feature.setdefault("properties", {})
        properties["classification"] = {
            "name": label,
            "colorRGB": hex_to_int_color(_resolve_color(color_mapping, label)),
        }
        label_counts[label] = label_counts.get(label, 0) + 1
        kept_features.append(feature)

    geojson_data["features"] = kept_features
    print(
        f"  Matched {matched}/{total_features} polygons; "
        f"wrote {len(kept_features)} after {'keeping' if keep_unassigned else 'dropping'} unassigned."
    )

    if stats is not None:
        stats.update(
            {
                "csv_rows": int(len(df)),
                "csv_rows_missing_label": int(missing_labels),
                "total_polygons": int(total_features),
                "matched_polygons": int(matched),
                "unassigned_polygons": int(total_features - matched),
                "written_polygons": int(len(kept_features)),
                "label_counts": label_counts,
                "color_mapping": dict(color_mapping),
            }
        )

    # 6. Save the annotated GeoJSON
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        if compact_output:
            json.dump(geojson_data, f, separators=(",", ":"))
        else:
            json.dump(geojson_data, f, indent=2)

    print(f"  Successfully saved annotated GeoJSON to: {output_path}")

    # 7. Generate PNG visualization if requested
    if generate_png:
        png_path = png_output_path or output_path.replace(".geojson", ".png")
        _generate_png_visualization(
            kept_features,
            png_path,
            color_mapping,
            mirror_y_axis,
            dpi,
            title=png_title,
            background_color=png_background,
            edge_color=png_edge_color,
            line_width=png_line_width,
            alpha=png_alpha,
        )

    return True


def generate_png_from_geojson(
    geojson_path: str,
    png_output_path: str,
    include_labels: list[str] | None = None,
    color_mapping: dict[str, str] | None = None,
    mirror_y_axis: bool = True,
    dpi: int = 150,
    title: str | None = None,
    background_color: str = "white",
    edge_color: str = "white",
    line_width: float = 0.5,
    alpha: float = 0.8,
    show_legend: bool = True,
) -> bool:
    """Generate a PNG visualization from a tagged GeoJSON file.

    Parameters
    ----------
    geojson_path : str
        Path to the tagged GeoJSON file.
    png_output_path : str
        Output path for the PNG file.
    include_labels : list[str] | None
        Optional list of phenotype labels to include. If None, includes all.
    color_mapping : dict[str, str] | None
        Mapping of phenotype labels to hex color codes. Missing labels get a
        color from the default palette.
    mirror_y_axis : bool
        Whether to mirror the Y-axis for image coordinate systems.

    Returns
    -------
    bool
        True if PNG generation succeeds, False otherwise.
    """
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to load GeoJSON file: {geojson_path} ({e})")
        return False

    features = geojson_data.get("features", []) if isinstance(geojson_data, dict) else []
    if not features:
        print(f"WARNING: No features found in GeoJSON: {geojson_path}")
        return False

    if include_labels:
        include_set = {str(v) for v in include_labels}
        features = [f for f in features if feature_label(f) in include_set]

    if not features:
        print(f"WARNING: No features matched selected labels in {geojson_path}")
        return False

    out_dir = os.path.dirname(png_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _generate_png_visualization(
        features,
        png_output_path,
        color_mapping,
        mirror_y_axis,
        dpi,
        title=title,
        background_color=background_color,
        edge_color=edge_color,
        line_width=line_width,
        alpha=alpha,
        show_legend=show_legend,
    )
    return True


def feature_label(feature: dict, default: str = DEFAULT_UNASSIGNED_LABEL) -> str:
    """Read the QuPath classification name off a GeoJSON feature."""
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
    classification = props.get("classification", {})
    if isinstance(classification, dict):
        return str(classification.get("name", default))
    if isinstance(classification, str) and classification:
        return classification
    return default


def _feature_rings(geom: dict) -> list:
    """Return every exterior ring in a Polygon / MultiPolygon geometry."""
    coordinates = geom.get("coordinates")
    if not coordinates:
        return []
    if geom["type"] == "Polygon":
        return [coordinates[0]]
    return [poly[0] for poly in coordinates if poly]


def _generate_png_visualization(
    features: list,
    png_path: str,
    color_mapping: dict[str, str] | None = None,
    mirror_y: bool = True,
    dpi: int = 150,
    title: str | None = None,
    background_color: str = "white",
    edge_color: str = "white",
    line_width: float = 0.5,
    alpha: float = 0.8,
    show_legend: bool = True,
) -> None:
    """Generate a PNG visualization of the segmented cells with solid colors.

    Parameters
    ----------
    features : list
        List of GeoJSON features with polygons and classification properties.
    png_path : str
        Path to save the PNG file.
    color_mapping : dict, optional
        Mapping from phenotype labels to hex colors. Labels missing from the
        mapping are colored from the default palette so nothing renders black
        by accident.
    mirror_y : bool, optional
        If True, mirror the y-axis coordinates (useful when GeoJSON uses image coordinates
        with origin at top-left). Default is True.
    """
    if not features:
        print("  No features to visualize, skipping PNG generation.")
        return

    # Every label present must have a color, otherwise the whole section renders
    # black and the legend comes out empty.
    color_mapping = build_color_mapping(
        (feature_label(f) for f in features), existing=dict(color_mapping or {})
    )

    # One vertex array per ring. Drawing 90k cells as individual Polygon patches
    # takes about two minutes; a single PolyCollection draws the same section in
    # a couple of seconds.
    vertices: list[np.ndarray] = []
    face_colors: list[str] = []
    phenotypes: list[str] = []

    for feature in features:
        geom = feature.get("geometry")
        if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
            continue

        phenotype = feature_label(feature)
        hex_color = _resolve_color(color_mapping, phenotype)

        for coords in _feature_rings(geom):
            if len(coords) < 3:
                continue
            ring = np.asarray(coords, dtype=float)
            if ring.ndim != 2 or ring.shape[1] < 2:
                continue
            vertices.append(ring[:, :2])
            face_colors.append(hex_color)
            phenotypes.append(phenotype)

    if not vertices:
        print("  No valid polygons found, skipping PNG generation.")
        return

    min_x = min(float(ring[:, 0].min()) for ring in vertices)
    max_x = max(float(ring[:, 0].max()) for ring in vertices)
    min_y = min(float(ring[:, 1].min()) for ring in vertices)
    max_y = max(float(ring[:, 1].max()) for ring in vertices)

    # Mirror y-coordinates if requested (for image coordinate systems where origin is top-left)
    if mirror_y:
        offset = max_y + min_y
        vertices = [np.column_stack((ring[:, 0], offset - ring[:, 1])) for ring in vertices]

    # Add 5% padding
    x_span = max(max_x - min_x, 1e-6)
    y_span = max(max_y - min_y, 1e-6)
    min_x -= x_span * 0.05
    max_x += x_span * 0.05
    min_y -= y_span * 0.05
    max_y += y_span * 0.05

    # Figure size maintaining aspect ratio, with extra width for the legend
    aspect_ratio = (max_x - min_x) / (max_y - min_y)
    if aspect_ratio > 1:
        fig_width = 16
        fig_height = 12 / aspect_ratio
    else:
        fig_width = 16 * aspect_ratio
        fig_height = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    plt.subplots_adjust(right=0.75)

    rgb_faces = []
    for hex_color in face_colors:
        try:
            rgb_faces.append(to_rgb(hex_color))
        except ValueError:
            rgb_faces.append(to_rgb("#000000"))

    ax.add_collection(
        PolyCollection(
            vertices,
            facecolors=rgb_faces,
            # A hairline around a 300 dpi section of half a million cells covers
            # the fill entirely, so width 0 has to mean "no outline" rather than
            # "thinnest available".
            edgecolors="none" if line_width <= 0 else edge_color,
            linewidths=max(float(line_width), 0.0),
            alpha=alpha,
            closed=True,
        )
    )

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(background_color)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if show_legend:
        legend_patches = []
        legend_labels = []
        for phenotype in sorted(set(phenotypes)):
            try:
                rgb_color = to_rgb(_resolve_color(color_mapping, phenotype))
            except ValueError:
                rgb_color = to_rgb("#000000")
            legend_patches.append(
                patches.Rectangle((0, 0), 1, 1, facecolor=rgb_color, edgecolor="black", linewidth=0.5)
            )
            legend_labels.append(phenotype)

        if legend_patches:
            legend = ax.legend(
                legend_patches,
                legend_labels,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="Phenotypes",
                title_fontsize=12,
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
            )
            legend.get_title().set_fontweight("bold")

    plt.tight_layout()

    out_dir = os.path.dirname(png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=background_color, edgecolor="none")
    plt.close(fig)

    print(f"  Successfully saved PNG visualization to: {png_path}")


def batch_annotate_geojson(
    csv_dir: str,
    geojson_dir: str,
    output_dir: str,
    phenotype_col: str,
    csv_pattern: str = "*.csv",
    geojson_pattern: str = "*_cell_boundaries.geojson",
    filter_criteria: dict | None = None,
    color_mapping: dict | None = None,
    png_output_dir: str | None = None,
    mirror_y_axis: bool = True,
    phenotype_suffix: str | None = None,
    mapping_dict: dict[str, str] | None = None,
    dpi: int = 150,
) -> None:
    """Find matching pairs of CSV and GeoJSON files and annotate them in batch.
    
    Parameters
    ----------
    csv_dir : str
        Directory containing CSV files with phenotype data.
    geojson_dir : str
        Directory containing GeoJSON files with cell boundaries.
    output_dir : str
        Directory to save annotated GeoJSON files.
    phenotype_col : str
        Column name in CSV files containing phenotype labels.
    csv_pattern : str, optional
        Glob pattern to match CSV files.
    geojson_pattern : str, optional
        Glob pattern to match GeoJSON files.
    filter_criteria : dict, optional
        Criteria to filter cells in the CSV.
    color_mapping : dict, optional
        Mapping from phenotype labels to hex colors.
    png_output_dir : str, optional
        Directory to save PNG visualizations. If None, PNGs are saved 
        alongside GeoJSON files but with .png extension.
    mirror_y_axis : bool, optional
        If True, mirror the y-axis coordinates in PNG output (useful for image coordinates).
        Default is True.
    phenotype_suffix : str, optional
        Suffix to add to output filenames to distinguish different phenotype methods.
        If None, no suffix is added.
    mapping_dict : dict[str, str], optional
        Dictionary mapping batch_id (from CSV) to batch_id (for GeoJSON matching).
        If provided, uses this for pairing instead of glob patterns.
    """
    if mapping_dict is not None:
        # Use provided mapping
        print("Using provided mapping dictionary for file pairing...")
        successful_annotations = 0
        for batch_id, batch_id in mapping_dict.items():
            # Find CSV file matching batch_id
            csv_files = glob.glob(os.path.join(csv_dir, f"{batch_id}*.csv"))
            if not csv_files:
                print(f"  - No CSV found for batch_id: {batch_id}")
                continue
            csv_path = csv_files[0]  # Take the first match
            
            # Find GeoJSON file matching batch_id
            geojson_files = glob.glob(os.path.join(geojson_dir, f"{batch_id}*.geojson"))
            if not geojson_files:
                print(f"  - No GeoJSON found for batch_id: {batch_id}")
                continue
            geojson_path = geojson_files[0]  # Take the first match
            
            # Use batch_id for output filename
            csv_basename = batch_id
            
            # Add phenotype suffix if provided
            if phenotype_suffix:
                output_filename = f"{csv_basename}_{phenotype_suffix}.geojson"
                png_filename = f"{csv_basename}_{phenotype_suffix}.png"
            else:
                output_filename = f"{csv_basename}.geojson"
                png_filename = f"{csv_basename}.png"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Determine PNG output path if png_output_dir is specified
            png_output_path = None
            if png_output_dir:
                png_output_path = os.path.join(png_output_dir, png_filename)
            
            print(f"  - Pairing: {os.path.basename(csv_path)} -> {os.path.basename(geojson_path)}")
            if phenotype_suffix:
                print(f"    Output: {output_filename} (suffix: {phenotype_suffix})")

            success = annotate_geojson_with_phenotypes(
                csv_path=csv_path,
                geojson_path=geojson_path,
                output_path=output_path,
                phenotype_col=phenotype_col,
                filter_criteria=filter_criteria,
                color_mapping=color_mapping,
                png_output_path=png_output_path,
                mirror_y_axis=mirror_y_axis,
                dpi=dpi,
            )
            if success:
                successful_annotations += 1
        
        print(f"\nBatch annotation complete. Successfully processed {successful_annotations} pairs.")
        return
    
    # Original glob-based method
    csv_files = glob.glob(os.path.join(csv_dir, csv_pattern))
    geojson_files = glob.glob(os.path.join(geojson_dir, geojson_pattern))

    if not csv_files:
        print(f"Warning: No CSV files found in '{csv_dir}' with pattern '{csv_pattern}'")
        return
    if not geojson_files:
        print(f"Warning: No GeoJSON files found in '{geojson_dir}' with pattern '{geojson_pattern}'")
        return

    def extract_csv_prefix(filename: str) -> str:
        """Extract prefix from CSV filename, handling both batch_id and circle patterns."""
        base = os.path.basename(filename).replace('.csv', '')
        # Check for circle pattern like P1_P3_roi_3_1_circle2 -> P1_P3_roi_3
        circle_match = re.match(r'^(.+_roi_\d+)_.*_circle\d*$', base)
        if circle_match:
            return circle_match.group(1)
        # Default to full basename without extension for regular batch files
        return base

    def extract_geojson_prefix(filename: str) -> str:
        """Extract prefix from GeoJSON filename."""
        return re.sub(r"\.ome.*", "", os.path.basename(filename))

    # Create maps from prefix to filepath
    csv_map = {}
    for f in csv_files:
        prefix = extract_csv_prefix(f)
        csv_map[prefix] = f

    geojson_map = {}
    for f in geojson_files:
        prefix = extract_geojson_prefix(f)
        geojson_map[prefix] = f

    print(f"Found {len(csv_map)} CSVs and {len(geojson_map)} GeoJSONs. Finding pairs...")
    
    # Quick validation: check if phenotype column exists in a sample CSV
    if csv_files:
        sample_csv = csv_files[0]
        try:
            import pandas as pd
            sample_df = pd.read_csv(sample_csv)
            if phenotype_col not in sample_df.columns:
                print(f"WARNING: Phenotype column '{phenotype_col}' not found in sample CSV: {os.path.basename(sample_csv)}")
                print(f"Available columns: {list(sample_df.columns)}")
        except Exception as e:
            print(f"Warning: Could not validate phenotype column in {os.path.basename(sample_csv)}: {e}")
    
    successful_annotations = 0
    
    for csv_prefix, csv_path in csv_map.items():
        if csv_prefix in geojson_map:
            geojson_path = geojson_map[csv_prefix]
            # Use original CSV filename (without .csv extension) for output
            csv_basename = os.path.basename(csv_path).replace('.csv', '')
            
            # Add phenotype suffix if provided
            if phenotype_suffix:
                output_filename = f"{csv_basename}_{phenotype_suffix}.geojson"
                png_filename = f"{csv_basename}_{phenotype_suffix}.png"
            else:
                output_filename = f"{csv_basename}.geojson"
                png_filename = f"{csv_basename}.png"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Determine PNG output path if png_output_dir is specified
            png_output_path = None
            if png_output_dir:
                png_output_path = os.path.join(png_output_dir, png_filename)
            
            print(f"  - Pairing: {os.path.basename(csv_path)} -> {os.path.basename(geojson_path)}")
            if phenotype_suffix:
                print(f"    Output: {output_filename} (suffix: {phenotype_suffix})")

            success = annotate_geojson_with_phenotypes(
                csv_path=csv_path,
                geojson_path=geojson_path,
                output_path=output_path,
                phenotype_col=phenotype_col,
                filter_criteria=filter_criteria,
                color_mapping=color_mapping,
                png_output_path=png_output_path,
                mirror_y_axis=mirror_y_axis,
                dpi=dpi,
            )
            if success:
                successful_annotations += 1
        else:
            print(f"  - No matching GeoJSON found for CSV prefix: {csv_prefix}")

    print(f"\nBatch annotation complete. Successfully processed {successful_annotations} pairs.")

def find_discrepant_circles(
    unified_csv_path: str,
    output_dir: str,
    pheno_cols: list[str],
    radius: float,
    target_count_range: tuple[int, int],
    disagree_range: tuple[float, float],
    epithelia_labels: list[str],
    epithelia_frac_range: tuple[float, float],
    epithelia_case_insensitive: bool,
    max_candidate_centers: int,
    random_state: int,
    max_circles_per_group: int,
    min_unique_phenotypes: int,
    allow_overlap: bool,
    overlap_threshold: float,
):
    """
    Find circular regions in a unified dataset that meet specific criteria for cell count,
    phenotype disagreement, and epithelial cell fraction.

    For each batch_id/tissue_roi group, this function searches for circular regions:
    - Containing a target number of cells.
    - Exhibiting a specified range of average pairwise disagreement among phenotype methods.
    - Containing a specific fraction of epithelial cells.

    Matching regions are saved as individual CSV files, and a summary file is generated.
    """
    from itertools import combinations
    from random import sample, seed
    import math

    # --- Helper Functions ---
    def _disagreement_metrics(sub: pd.DataFrame, pairwise_combinations: list) -> dict:
        metrics = {}
        disagreements = []
        for a, b in pairwise_combinations:
            col_a = sub[a].astype(str)
            col_b = sub[b].astype(str)
            mask_valid = col_a.notna() & col_b.notna()
            if mask_valid.sum() == 0:
                frac_disagree = np.nan
            else:
                equal = (col_a[mask_valid] == col_b[mask_valid]).sum()
                total = mask_valid.sum()
                frac_match = equal / total
                frac_disagree = 1 - frac_match
            metrics[f'{a}__vs__{b}_disagree'] = frac_disagree
            if not math.isnan(frac_disagree):
                disagreements.append(frac_disagree)
        metrics['avg_disagree'] = np.mean(disagreements) if disagreements else np.nan
        return metrics

    def _epithelia_fractions(sub: pd.DataFrame, pheno_cols_list: list) -> tuple[dict, float, bool]:
        fracs = {}
        acceptable = False
        labels_to_check = epithelia_labels
        labels_lower = set()  # Initialize to avoid lint warning
        if epithelia_case_insensitive:
            labels_lower = {l.lower() for l in labels_to_check}

        for col in pheno_cols_list:
            series = sub[col].dropna().astype(str)
            if epithelia_case_insensitive:
                match = series.str.lower().isin(labels_lower).sum()
            else:
                match = series.isin(labels_to_check).sum()
            denom = len(series)
            frac = match / denom if denom else 0.0
            fracs[f'epithelia_frac__{col}'] = frac
            if epithelia_frac_range[0] <= frac <= epithelia_frac_range[1]:
                acceptable = True
        max_frac = max(fracs.values()) if fracs else 0.0
        return fracs, max_frac, acceptable

    def _choose_candidate_indices(n: int) -> np.ndarray:
        if n <= max_candidate_centers:
            return np.arange(n)
        return np.array(sample(range(n), max_candidate_centers))

    def _derive_prefix(batch_id: str) -> str:
        m = re.search(r'(.+?roi_\\d+)', batch_id)
        if m:
            return m.group(1)
        parts = batch_id.split('_')
        return '_'.join(parts[:3]) if len(parts) >= 3 else batch_id

    def _circle_search(group_df: pd.DataFrame, pairwise_combinations: list) -> list:
        coords = group_df[['x_centroid', 'y_centroid']].to_numpy(dtype=float)
        n = len(coords)
        if n < target_count_range[0]:
            return []

        cand_idx = _choose_candidate_indices(n)
        accepted = []
        used_indices_sets = []
        r2 = radius * radius
        ideal_count = sum(target_count_range) / 2
        ideal_disagree = sum(disagree_range) / 2

        for idx in cand_idx:
            center = coords[idx]
            deltas = coords - center
            dist2 = (deltas[:, 0]**2 + deltas[:, 1]**2)
            inside_mask = dist2 <= r2
            count_inside = int(inside_mask.sum())

            if not (target_count_range[0] <= count_inside <= target_count_range[1]):
                continue

            sub = group_df.loc[inside_mask]
            if len({p for c in pheno_cols for p in sub[c].dropna().astype(str).unique()}) < min_unique_phenotypes:
                continue

            epi_fracs, max_epi_frac, epi_ok = _epithelia_fractions(sub, pheno_cols)
            if not epi_ok:
                continue

            mets = _disagreement_metrics(sub, pairwise_combinations)
            avg_dis = mets['avg_disagree']
            if math.isnan(avg_dis) or not (disagree_range[0] <= avg_dis <= disagree_range[1]):
                continue

            indices_inside = set(sub.index.tolist())
            if not allow_overlap:
                if any(len(indices_inside & used) / len(indices_inside) > overlap_threshold for used in used_indices_sets):
                    continue
            
            score = (abs(count_inside - ideal_count) / ideal_count) + \
                    (abs(avg_dis - ideal_disagree) / ideal_disagree) + \
                    (abs(max_epi_frac - sum(epithelia_frac_range)/2) / (sum(epithelia_frac_range)/2))

            accepted.append({
                'center_x': center[0], 'center_y': center[1], 'count': count_inside,
                'avg_disagree': avg_dis, 'max_epithelia_frac': max_epi_frac,
                'score': score, 'indices': indices_inside, **mets, **epi_fracs
            })
            used_indices_sets.append(indices_inside)

        accepted.sort(key=lambda d: d['score'])
        return accepted[:max_circles_per_group] if max_circles_per_group else accepted

    # --- Main Execution ---
    seed(random_state)
    np.random.seed(random_state)

    print("Loading unified CSV...")
    df = pd.read_csv(unified_csv_path)
    print(f"Loaded {len(df):,} rows")

    required_cols = ['batch_id', 'x_centroid', 'y_centroid'] + pheno_cols
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"Unified file missing required columns: {set(required_cols) - set(df.columns)}")

    tissue_roi_col = 'tissue_roi'
    if tissue_roi_col not in df.columns or pd.isna(df[tissue_roi_col]).all():
        df[tissue_roi_col] = df['batch_id'].astype(str).str.extract(r'roi_(\\d+)')[0]

    if tissue_roi_col not in df.columns:
        raise ValueError("tissue_roi column could not be inferred or is missing.")

    results_meta = []
    output_files = []
    pairwise_combinations = list(combinations(pheno_cols, 2))
    grouped = df.groupby(['batch_id', tissue_roi_col], dropna=False)
    print(f"Processing {len(grouped)} (batch_id, tissue_roi) groups...")

    for (batch_id, tissue_roi), gdf in grouped:
        if len(gdf) < target_count_range[0]:
            continue
        
        circles = _circle_search(gdf, pairwise_combinations)
        if not circles:
            continue

        prefix = _derive_prefix(str(batch_id))
        for i, circ in enumerate(circles, start=1):
            circ_df = gdf.loc[list(circ['indices'])].copy()
            circle_id = f"{prefix}_tissueROI{tissue_roi}_circle{i}"
            circ_df['circle_id'] = circle_id
            out_name = f"{prefix}_{tissue_roi}_circle{i}.csv"
            out_path = os.path.join(output_dir, out_name)
            circ_df.to_csv(out_path, index=False)
            output_files.append(out_path)

            meta_row = {
                'batch_id': batch_id, 'tissue_roi': tissue_roi, 'circle_name': circle_id,
                'file': str(out_path), 'center_x': circ['center_x'], 'center_y': circ['center_y'],
                'cell_count': circ['count'], 'avg_disagree': circ['avg_disagree'],
                'max_epithelia_frac': circ['max_epithelia_frac'], 'score': circ['score'],
                **{k: v for k, v in circ.items() if k.endswith('_disagree') or k.startswith('epithelia_frac__')}
            }
            results_meta.append(meta_row)

    if results_meta:
        meta_df = pd.DataFrame(results_meta).sort_values('score')
        summary_path = os.path.join(output_dir, 'circle_regions_summary.csv')
        meta_df.to_csv(summary_path, index=False)
        print(f"\\nFound {len(results_meta)} circle regions. Summary: {summary_path}")
    else:
        print("\\nNo circle regions found meeting the criteria.")

    print("Output files:")
    for p in output_files[:10]:
        print(f" - {p}")
    if len(output_files) > 10:
        print(f"... ({len(output_files)-10} more)")
    
    print("Done.")
