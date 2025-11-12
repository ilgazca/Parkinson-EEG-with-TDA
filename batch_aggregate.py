#!/usr/bin/env python3
"""
Batch Aggregation Script for All Patients

Runs aggregate_features.py on all patient folders.

Usage:
    python batch_aggregate.py [options]

Example:
    python batch_aggregate.py --method full --include-variability
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_patient_folders(base_dir='.'):
    """
    Find all patient folders containing *_all_features*.pkl files.

    Args:
        base_dir: Base directory to search

    Returns:
        List of patient folder paths
    """
    base_path = Path(base_dir)
    patient_folders = []

    # Look for directories containing *_all_features*.pkl files (includes holdL/holdR suffixes)
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this folder has feature files
            feature_files = list(item.glob("*_all_features*.pkl"))
            if feature_files:
                patient_folders.append(item)

    return sorted(patient_folders)


def run_aggregation(patient_folder, method='mean', include_variability=False,
                   distance_metric='wasserstein', output='aggregated', quiet=False):
    """
    Run aggregate_features.py on a single patient folder.

    Args:
        patient_folder: Path to patient folder
        method: Aggregation method
        include_variability: Include variability features
        distance_metric: Distance metric for diagrams
        output: Output filename suffix
        quiet: Suppress output

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'aggregate_features.py',
        str(patient_folder),
        '--method', method,
        '--distance-metric', distance_metric,
        '--output', output
    ]

    if include_variability:
        cmd.append('--include-variability')

    if quiet:
        cmd.append('--quiet')

    try:
        result = subprocess.run(cmd, check=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to aggregate {patient_folder}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch aggregate features for all patients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all patients with default settings
  python batch_aggregate.py

  # Full statistics with variability
  python batch_aggregate.py --method full --include-variability

  # Process only specific patients
  python batch_aggregate.py --patients i4oK0F QZTsn6
        """
    )

    parser.add_argument('--method', choices=['mean', 'median', 'full'], default='mean',
                        help='Aggregation method (default: mean)')
    parser.add_argument('--include-variability', action='store_true',
                        help='Include variability features')
    parser.add_argument('--distance-metric', choices=['wasserstein', 'bottleneck'], default='wasserstein',
                        help='Distance metric for diagram aggregation (default: wasserstein)')
    parser.add_argument('--output', default='aggregated',
                        help='Output filename suffix (default: aggregated)')
    parser.add_argument('--patients', nargs='+',
                        help='Specific patient folders to process (default: all)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    parser.add_argument('--base-dir', default='.',
                        help='Base directory to search for patient folders (default: current directory)')

    args = parser.parse_args()

    # Find patient folders
    if args.patients:
        # Use specified patients
        patient_folders = [Path(args.base_dir) / p for p in args.patients]
        # Validate they exist
        patient_folders = [p for p in patient_folders if p.exists() and p.is_dir()]
    else:
        # Find all patient folders
        patient_folders = find_patient_folders(args.base_dir)

    if not patient_folders:
        print("ERROR: No patient folders found")
        sys.exit(1)

    print(f"Found {len(patient_folders)} patient folder(s) to process:")
    for folder in patient_folders:
        print(f"  - {folder.name}")

    print(f"\nAggregation settings:")
    print(f"  Method: {args.method}")
    print(f"  Include variability: {args.include_variability}")
    print(f"  Distance metric: {args.distance_metric}")
    print(f"  Output suffix: {args.output}")

    # Process each patient
    success_count = 0
    failed_patients = []

    for i, patient_folder in enumerate(patient_folders, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(patient_folders)}: {patient_folder.name}")
        print(f"{'='*80}")

        success = run_aggregation(
            patient_folder=patient_folder,
            method=args.method,
            include_variability=args.include_variability,
            distance_metric=args.distance_metric,
            output=args.output,
            quiet=args.quiet
        )

        if success:
            success_count += 1
        else:
            failed_patients.append(patient_folder.name)

    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH AGGREGATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  Successful: {success_count}/{len(patient_folders)}")
    if failed_patients:
        print(f"  Failed: {len(failed_patients)}")
        print(f"    - " + "\n    - ".join(failed_patients))


if __name__ == "__main__":
    main()
