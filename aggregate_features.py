#!/usr/bin/env python3
"""
Aggregate Multiple Slices into Single Representative Features

This script combines features from multiple time slices per condition into single
representative feature vectors/matrices for downstream analysis.

Usage:
    python aggregate_features.py <patient_folder> [options]

Example:
    python aggregate_features.py ./i4oK0F/ --method mean
    python aggregate_features.py ./i4oK0F/ --method full --include-variability
"""

import argparse
import numpy as np
import pickle
import os
from pathlib import Path
import sys

# Import TDA libraries
try:
    from gtda.diagrams import PairwiseDistance, Scaler
    from eeg_utils import pad_diagrams
    GTDA_AVAILABLE = True
except ImportError:
    print("WARNING: gtda not available. Medoid computation for diagrams will be disabled.")
    GTDA_AVAILABLE = False


def select_representative_slice_medoid(diagrams, metric='wasserstein'):
    """
    Select the diagram that is most representative (medoid).
    This is Solution 5 from ANALYSIS_METHODOLOGY.md

    Args:
        diagrams: List of persistence diagrams
        metric: Distance metric ('wasserstein' or 'bottleneck')

    Returns:
        Tuple of (representative_diagram, medoid_index, mean_distance)
    """
    if not GTDA_AVAILABLE:
        print("WARNING: gtda not available, returning first diagram")
        return diagrams[0], 0, 0.0

    if len(diagrams) == 1:
        return diagrams[0], 0, 0.0

    try:
        # Pad and scale
        diagrams_2d = [d[0] if len(d.shape) == 3 else d for d in diagrams]
        diagrams_padded = pad_diagrams(diagrams_2d)

        scaler = Scaler(metric=metric)
        diagrams_scaled = scaler.fit_transform(diagrams_padded)

        # Compute pairwise distances
        pwise = PairwiseDistance(metric=metric)
        dist_matrix = pwise.fit_transform(diagrams_scaled)[0]

        # Find medoid
        sum_distances = np.sum(dist_matrix, axis=1)
        medoid_idx = np.argmin(sum_distances)
        mean_dist = sum_distances[medoid_idx] / len(diagrams)

        return diagrams[medoid_idx], int(medoid_idx), float(mean_dist)

    except Exception as e:
        print(f"WARNING: Error computing medoid ({e}), returning first diagram")
        return diagrams[0], 0, 0.0


def aggregate_scalar_features_mean(values):
    """
    Solution 1: Simple mean and std for scalar features.

    Args:
        values: List or array of scalar values

    Returns:
        Dictionary with mean and std
    """
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


def aggregate_scalar_features_full_stats(values):
    """
    Solution 2: Full statistical summary (mean, std, min, max, median, range).

    Args:
        values: List or array of scalar values

    Returns:
        Dictionary with multiple statistics
    """
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'range': float(np.max(values) - np.min(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
    }


def aggregate_array_features(arrays, method='mean'):
    """
    Aggregate array-valued features (Betti curves, landscapes, heat kernels).

    Args:
        arrays: List of numpy arrays (same shape)
        method: 'mean', 'median', or 'full'

    Returns:
        Dictionary with aggregated arrays
    """
    arrays = np.array(arrays)

    if method == 'mean':
        return {
            'mean': np.mean(arrays, axis=0),
            'std': np.std(arrays, axis=0)
        }
    elif method == 'median':
        return {
            'median': np.median(arrays, axis=0),
            'mad': np.median(np.abs(arrays - np.median(arrays, axis=0, keepdims=True)), axis=0)  # Median absolute deviation
        }
    elif method == 'full':
        return {
            'mean': np.mean(arrays, axis=0),
            'std': np.std(arrays, axis=0),
            'median': np.median(arrays, axis=0),
            'min': np.min(arrays, axis=0),
            'max': np.max(arrays, axis=0)
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_variability_features(values):
    """
    Solution 8: Extract variability measures as features.

    Args:
        values: List or array of scalar values

    Returns:
        Dictionary with variability metrics
    """
    values = np.array(values)
    mean_val = np.mean(values)

    variability = {
        'std': float(np.std(values)),
        'range': float(np.max(values) - np.min(values)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
    }

    # Coefficient of variation (only if mean is non-zero)
    if abs(mean_val) > 1e-10:
        variability['cv'] = float(np.std(values) / abs(mean_val))

    return variability


def aggregate_summary_statistics(stats_list, method='mean', include_variability=False):
    """
    Aggregate summary statistics dictionaries across slices.

    Args:
        stats_list: List of dictionaries (one per slice) with keys like 'H0_count', 'H1_mean_lifespan', etc.
        method: 'mean', 'median', or 'full'
        include_variability: Whether to include variability features

    Returns:
        Dictionary with aggregated statistics
    """
    if not stats_list:
        return {}

    aggregated = {}

    # Get all keys from first slice
    keys = stats_list[0].keys()

    for key in keys:
        # Extract values for this key across all slices
        values = [s[key] for s in stats_list if key in s]

        if method == 'mean':
            agg = aggregate_scalar_features_mean(values)
            aggregated[f'{key}_mean'] = agg['mean']
            aggregated[f'{key}_std'] = agg['std']

        elif method == 'median':
            values_arr = np.array(values)
            aggregated[f'{key}_median'] = float(np.median(values_arr))
            aggregated[f'{key}_mad'] = float(np.median(np.abs(values_arr - np.median(values_arr))))

        elif method == 'full':
            agg = aggregate_scalar_features_full_stats(values)
            for stat_name, stat_value in agg.items():
                aggregated[f'{key}_{stat_name}'] = stat_value

        # Add variability features if requested
        if include_variability:
            var_features = extract_variability_features(values)
            for var_name, var_value in var_features.items():
                aggregated[f'{key}_var_{var_name}'] = var_value

    return aggregated


def aggregate_condition(all_features, hemisphere, condition, method='mean', include_variability=False):
    """
    Aggregate all features for a specific hemisphere-condition combination.

    Args:
        all_features: Dictionary from the *_all_features.pkl file
        hemisphere: 'left' or 'right'
        condition: 'hold' or 'resting'
        method: Aggregation method ('mean', 'median', 'full')
        include_variability: Whether to include variability features

    Returns:
        Dictionary with aggregated features
    """
    prefix = f'{hemisphere}_{condition}'
    aggregated = {}

    print(f"  Aggregating {prefix}...")

    # 1. Persistence Entropy
    if f'{prefix}_pe' in all_features:
        pe_values = all_features[f'{prefix}_pe']
        print(f"    - Persistence Entropy: {len(pe_values)} slices")

        if method == 'mean':
            agg = aggregate_scalar_features_mean(pe_values)
            aggregated['persistence_entropy_mean'] = agg['mean']
            aggregated['persistence_entropy_std'] = agg['std']
        elif method == 'median':
            pe_arr = np.array(pe_values)
            aggregated['persistence_entropy_median'] = float(np.median(pe_arr))
            aggregated['persistence_entropy_mad'] = float(np.median(np.abs(pe_arr - np.median(pe_arr))))
        elif method == 'full':
            agg = aggregate_scalar_features_full_stats(pe_values)
            for stat_name, stat_value in agg.items():
                aggregated[f'persistence_entropy_{stat_name}'] = stat_value

        if include_variability:
            var_features = extract_variability_features(pe_values)
            for var_name, var_value in var_features.items():
                aggregated[f'persistence_entropy_var_{var_name}'] = var_value

    # 2. Summary Statistics
    if f'{prefix}_stats' in all_features:
        stats_list = all_features[f'{prefix}_stats']
        print(f"    - Summary Statistics: {len(stats_list)} slices")

        stats_agg = aggregate_summary_statistics(stats_list, method=method, include_variability=include_variability)
        aggregated.update(stats_agg)

    # 3. Persistence Landscapes
    if f'{prefix}_pl' in all_features:
        pl_values = all_features[f'{prefix}_pl']
        print(f"    - Persistence Landscapes: {len(pl_values)} slices")

        pl_agg = aggregate_array_features(pl_values, method=method)
        for key, value in pl_agg.items():
            aggregated[f'persistence_landscape_{key}'] = value

    # 4. Betti Curves
    if f'{prefix}_bc' in all_features:
        bc_values = all_features[f'{prefix}_bc']
        print(f"    - Betti Curves: {len(bc_values)} slices")

        bc_agg = aggregate_array_features(bc_values, method=method)
        for key, value in bc_agg.items():
            aggregated[f'betti_curve_{key}'] = value

    # 5. Heat Kernel
    if f'{prefix}_hk' in all_features:
        hk_values = all_features[f'{prefix}_hk']
        print(f"    - Heat Kernel: {len(hk_values)} slices")

        hk_agg = aggregate_array_features(hk_values, method=method)
        for key, value in hk_agg.items():
            aggregated[f'heat_kernel_{key}'] = value

    return aggregated


def aggregate_persistence_diagrams(patient_folder, med_state, hemisphere, condition, metric='wasserstein'):
    """
    Aggregate persistence diagrams using medoid selection.

    Args:
        patient_folder: Path to patient directory
        med_state: 'medOn' or 'medOff'
        hemisphere: 'left' or 'right'
        condition: 'hold' or 'resting'
        metric: Distance metric for medoid computation

    Returns:
        Dictionary with representative diagram and metadata
    """
    prefix = f"{med_state}_{hemisphere}_{condition}"
    diagrams_file = Path(patient_folder) / f"{prefix}_diagrams.pkl"

    if not diagrams_file.exists():
        print(f"    WARNING: Diagrams file not found: {diagrams_file}")
        return {}

    # Load diagrams
    with open(diagrams_file, 'rb') as f:
        diagrams = pickle.load(f)

    print(f"    - Persistence Diagrams: {len(diagrams)} slices")

    # Select representative (medoid)
    representative, medoid_idx, mean_dist = select_representative_slice_medoid(diagrams, metric=metric)

    return {
        'persistence_diagram': representative,
        'medoid_index': medoid_idx,
        'mean_distance_to_slices': mean_dist,
        'n_slices': len(diagrams)
    }


def aggregate_patient_features(patient_folder, method='mean', include_variability=False,
                               distance_metric='wasserstein', verbose=True):
    """
    Aggregate all features for a single patient across all conditions.

    Args:
        patient_folder: Path to patient directory
        method: Aggregation method ('mean', 'median', 'full')
        include_variability: Whether to include variability features
        distance_metric: Metric for diagram aggregation
        verbose: Print progress

    Returns:
        Dictionary with all aggregated features
    """
    patient_folder = Path(patient_folder)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Aggregating features for patient: {patient_folder.name}")
        print(f"{'='*80}")

    aggregated_data = {}

    # Find all *_all_features.pkl files
    feature_files = list(patient_folder.glob("*_all_features.pkl"))

    if not feature_files:
        print(f"ERROR: No *_all_features.pkl files found in {patient_folder}")
        return None

    for feature_file in feature_files:
        # Extract medication state from filename (e.g., 'medOff_all_features.pkl' -> 'medOff')
        med_state = feature_file.stem.replace('_all_features', '')

        if verbose:
            print(f"\nProcessing {med_state}...")

        # Load features
        with open(feature_file, 'rb') as f:
            all_features = pickle.load(f)

        med_state_data = {}

        # Aggregate for each hemisphere and condition
        for hemisphere in ['left', 'right']:
            for condition in ['hold', 'resting']:
                # Aggregate non-diagram features
                agg_features = aggregate_condition(
                    all_features,
                    hemisphere,
                    condition,
                    method=method,
                    include_variability=include_variability
                )

                # Aggregate persistence diagrams
                diagram_features = aggregate_persistence_diagrams(
                    patient_folder,
                    med_state,
                    hemisphere,
                    condition,
                    metric=distance_metric
                )

                # Combine
                combined = {
                    'hemisphere': hemisphere,
                    'condition': condition,
                    **agg_features,
                    **diagram_features
                }

                med_state_data[f'{hemisphere}_{condition}'] = combined

        # Add distance matrices if they exist (these are already aggregated)
        if 'left_wasserstein' in all_features:
            med_state_data['left_wasserstein'] = all_features['left_wasserstein']
        if 'right_wasserstein' in all_features:
            med_state_data['right_wasserstein'] = all_features['right_wasserstein']
        if 'left_bottleneck' in all_features:
            med_state_data['left_bottleneck'] = all_features['left_bottleneck']
        if 'right_bottleneck' in all_features:
            med_state_data['right_bottleneck'] = all_features['right_bottleneck']

        aggregated_data[med_state] = med_state_data

    return aggregated_data


def save_aggregated_features(aggregated_data, patient_folder, suffix='aggregated'):
    """
    Save aggregated features to pickle file.

    Args:
        aggregated_data: Dictionary with aggregated features
        patient_folder: Path to patient directory
        suffix: Suffix for output filename
    """
    output_file = Path(patient_folder) / f"{suffix}_features.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(aggregated_data, f)

    print(f"\n{'='*80}")
    print(f"Aggregated features saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary
    print(f"\nSummary:")
    for med_state, med_data in aggregated_data.items():
        print(f"\n  {med_state}:")
        for key, value in med_data.items():
            if isinstance(value, dict):
                n_features = len([k for k in value.keys() if not k in ['hemisphere', 'condition']])
                print(f"    - {key}: {n_features} features")
            else:
                print(f"    - {key}: {type(value).__name__} shape {getattr(value, 'shape', 'N/A')}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multiple time slices into single representative features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aggregation Methods:
  mean    : Simple mean and std (fastest, recommended for most cases)
  median  : Median and MAD (robust to outliers)
  full    : Mean, std, min, max, median, range, IQR (most comprehensive)

Examples:
  # Basic usage with mean aggregation
  python aggregate_features.py ./i4oK0F/

  # Full statistics with variability features
  python aggregate_features.py ./i4oK0F/ --method full --include-variability

  # Robust aggregation using median
  python aggregate_features.py ./i4oK0F/ --method median

  # Custom output filename
  python aggregate_features.py ./i4oK0F/ --output my_aggregated_features
        """
    )

    parser.add_argument('patient_folder', help='Path to patient folder containing *_all_features.pkl files')
    parser.add_argument('--method', choices=['mean', 'median', 'full'], default='mean',
                        help='Aggregation method (default: mean)')
    parser.add_argument('--include-variability', action='store_true',
                        help='Include variability features (std, range, IQR, CV)')
    parser.add_argument('--distance-metric', choices=['wasserstein', 'bottleneck'], default='wasserstein',
                        help='Distance metric for diagram aggregation (default: wasserstein)')
    parser.add_argument('--output', default='aggregated',
                        help='Output filename suffix (default: aggregated)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    # Validate patient folder
    if not os.path.exists(args.patient_folder):
        print(f"ERROR: Patient folder not found: {args.patient_folder}")
        sys.exit(1)

    # Aggregate features
    aggregated_data = aggregate_patient_features(
        patient_folder=args.patient_folder,
        method=args.method,
        include_variability=args.include_variability,
        distance_metric=args.distance_metric,
        verbose=not args.quiet
    )

    if aggregated_data is None:
        print("ERROR: Failed to aggregate features")
        sys.exit(1)

    # Save results
    output_file = save_aggregated_features(
        aggregated_data,
        args.patient_folder,
        suffix=args.output
    )

    print(f"\n✓ Aggregation complete!")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
