#!/usr/bin/env python3
"""
Feature Extraction Script for Parkinson's EEG/LFP Data
This script performs the complete TDA pipeline on LFP data.

Usage:
    python feature_extraction.py <mat_file> <event_times_file> <output_folder> [options]

Example:
    python feature_extraction.py data.mat events.txt ./output --med-state medOff --task hold
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import decimate
import pickle
import os
from pathlib import Path
import sys

# Import custom utilities
from eeg_utils import (
    mat_to_dataframe,
    butter_bandpass_filter,
    fit_embedder,
    extract_features,
    pad_diagrams
)

# Import TDA libraries
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (
    PersistenceEntropy,
    PersistenceLandscape,
    BettiCurve,
    HeatKernel,
    PairwiseDistance,
    Scaler
)


def load_event_times(event_file):
    """
    Load event times from a text file.

    Args:
        event_file (str): Path to event times file

    Returns:
        dict: Dictionary containing 'hold' and 'resting' event times as lists
    """
    print(f"\n--- Loading Event Times from {event_file} ---")

    # Try reading as simple list of times
    try:
        with open(event_file, 'r') as f:
            lines = f.readlines()

        # Parse the file - assuming format with labels
        events = {'hold': [], 'resting': []}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                time = float(parts[0])
                label = parts[1].lower()

                if 'hold' in label:
                    events['hold'].append(time)
                elif 'rest' in label:
                    events['resting'].append(time)
            else:
                # If just a number, treat as hold event
                try:
                    events['hold'].append(float(line))
                except ValueError:
                    continue

        print(f"Found {len(events['hold'])} hold events and {len(events['resting'])} resting events")
        return events

    except Exception as e:
        print(f"Error loading event times: {e}")
        sys.exit(1)


def slice_signal_by_events(signal, event_times, fs, slice_length=60):
    """
    Slice a signal based on event times.

    Args:
        signal (np.ndarray): The signal to slice
        event_times (list): List of event start times in seconds
        fs (float): Sampling frequency
        slice_length (float): Length of each slice in seconds

    Returns:
        list: List of signal slices
    """
    slices = []

    for start_time_sec in event_times:
        start_index = int(start_time_sec * fs)
        end_index = int((start_time_sec + slice_length) * fs)

        # Ensure indices are within bounds
        if start_index < 0:
            start_index = 0
        if end_index > len(signal):
            end_index = len(signal)

        slice_data = signal[start_index:end_index]
        slices.append(slice_data)

    return slices


def process_pipeline(mat_file, event_times_file, output_folder, args):
    """
    Main processing pipeline for TDA feature extraction.

    Args:
        mat_file (str): Path to .mat data file
        event_times_file (str): Path to event times file
        output_folder (str): Directory to save output files
        args: Command-line arguments
    """

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PARKINSON'S LFP TDA FEATURE EXTRACTION PIPELINE")
    print("="*80)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[1/8] Loading LFP data...")
    result = mat_to_dataframe(mat_file)
    if result is None:
        print("ERROR: Failed to load data file")
        sys.exit(1)

    df, left_lfp, right_lfp, left_name, right_name = result
    original_fs = 2000  # Default sampling rate

    # ========================================================================
    # STEP 2: Bandpass Filtering
    # ========================================================================
    print(f"\n[2/8] Applying bandpass filter ({args.lowcut}-{args.highcut} Hz)...")
    left_filtered = butter_bandpass_filter(
        data=left_lfp,
        lowcut=args.lowcut,
        highcut=args.highcut,
        fs=original_fs,
        order=5
    )

    right_filtered = butter_bandpass_filter(
        data=right_lfp,
        lowcut=args.lowcut,
        highcut=args.highcut,
        fs=original_fs,
        order=5
    )
    print(f"✓ Filtered {left_name} and {right_name}")

    # ========================================================================
    # STEP 3: Downsampling
    # ========================================================================
    print(f"\n[3/8] Downsampling from {original_fs} Hz to {args.target_fs} Hz...")

    if original_fs % args.target_fs != 0:
        print(f"WARNING: {original_fs} is not evenly divisible by {args.target_fs}")

    q = original_fs // args.target_fs
    left_downsampled = decimate(x=left_filtered, q=q, ftype="fir", zero_phase=True)
    right_downsampled = decimate(x=right_filtered, q=q, ftype="fir", zero_phase=True)

    print(f"✓ Downsampled: {len(left_filtered)} → {len(left_downsampled)} samples")

    # ========================================================================
    # STEP 4: Load Event Times and Slice
    # ========================================================================
    print(f"\n[4/8] Loading event times and slicing data...")
    events = load_event_times(event_times_file)

    # Slice the signals
    left_hold = slice_signal_by_events(left_downsampled, events['hold'], args.target_fs, args.slice_length)
    right_hold = slice_signal_by_events(right_downsampled, events['hold'], args.target_fs, args.slice_length)
    left_resting = slice_signal_by_events(left_downsampled, events['resting'], args.target_fs, args.slice_length)
    right_resting = slice_signal_by_events(right_downsampled, events['resting'], args.target_fs, args.slice_length)

    print(f"✓ Created slices: {len(left_hold)} hold, {len(left_resting)} resting per channel")

    # Save sliced time series if requested
    if args.save_timeseries:
        print("\nSaving sliced time series...")
        pickle.dump(left_hold, open(os.path.join(output_folder, f"{args.prefix}_left_hold.pkl"), "wb"))
        pickle.dump(right_hold, open(os.path.join(output_folder, f"{args.prefix}_right_hold.pkl"), "wb"))
        pickle.dump(left_resting, open(os.path.join(output_folder, f"{args.prefix}_left_resting.pkl"), "wb"))
        pickle.dump(right_resting, open(os.path.join(output_folder, f"{args.prefix}_right_resting.pkl"), "wb"))

    # ========================================================================
    # STEP 5: Takens Embedding
    # ========================================================================
    print(f"\n[5/8] Computing Takens embeddings...")
    print(f"Parameters: max_dimension={args.max_embedding_dim}, max_time_delay={args.max_time_delay}")

    embedder = SingleTakensEmbedding(
        parameters_type="search",
        time_delay=args.max_time_delay,
        dimension=args.max_embedding_dim,
        stride=1,
        n_jobs=-1
    )

    # Compute embeddings for all slices
    left_hold_embeddings = []
    left_resting_embeddings = []
    right_hold_embeddings = []
    right_resting_embeddings = []

    print("\nEmbedding left hold slices...")
    for i, slice_data in enumerate(left_hold):
        print(f"  Slice {i+1}/{len(left_hold)}...", end=" ")
        emb = fit_embedder(embedder, slice_data, verbose=False)
        left_hold_embeddings.append(emb)
        print(f"✓ Shape: {emb.shape}")

    print("\nEmbedding left resting slices...")
    for i, slice_data in enumerate(left_resting):
        print(f"  Slice {i+1}/{len(left_resting)}...", end=" ")
        emb = fit_embedder(embedder, slice_data, verbose=False)
        left_resting_embeddings.append(emb)
        print(f"✓ Shape: {emb.shape}")

    print("\nEmbedding right hold slices...")
    for i, slice_data in enumerate(right_hold):
        print(f"  Slice {i+1}/{len(right_hold)}...", end=" ")
        emb = fit_embedder(embedder, slice_data, verbose=False)
        right_hold_embeddings.append(emb)
        print(f"✓ Shape: {emb.shape}")

    print("\nEmbedding right resting slices...")
    for i, slice_data in enumerate(right_resting):
        print(f"  Slice {i+1}/{len(right_resting)}...", end=" ")
        emb = fit_embedder(embedder, slice_data, verbose=False)
        right_resting_embeddings.append(emb)
        print(f"✓ Shape: {emb.shape}")

    # ========================================================================
    # STEP 6: Persistent Homology
    # ========================================================================
    print(f"\n[6/8] Computing persistent homology (H0, H1, H2, H3)...")
    print(f"Using first {args.embedding_subset_size} points from each embedding")

    homology_dims = [0, 1, 2, 3]
    persistence = VietorisRipsPersistence(homology_dimensions=homology_dims, n_jobs=-1)

    def compute_diagrams(embeddings, name):
        diagrams = []
        print(f"\nProcessing {name}...")
        for i, embedding in enumerate(embeddings):
            print(f"  Diagram {i+1}/{len(embeddings)}...", end=" ")
            embedding_subset = embedding[:args.embedding_subset_size]
            embedding_3d = embedding_subset[None, :, :]
            diagram = persistence.fit_transform(embedding_3d)
            diagrams.append(diagram)
            print("✓")
        return diagrams

    left_hold_diagrams = compute_diagrams(left_hold_embeddings, "Left Hold")
    left_resting_diagrams = compute_diagrams(left_resting_embeddings, "Left Resting")
    right_hold_diagrams = compute_diagrams(right_hold_embeddings, "Right Hold")
    right_resting_diagrams = compute_diagrams(right_resting_embeddings, "Right Resting")

    # Save persistence diagrams
    print("\nSaving persistence diagrams...")
    pickle.dump(left_hold_diagrams, open(os.path.join(output_folder, f"{args.prefix}_left_hold_diagrams.pkl"), "wb"))
    pickle.dump(left_resting_diagrams, open(os.path.join(output_folder, f"{args.prefix}_left_resting_diagrams.pkl"), "wb"))
    pickle.dump(right_hold_diagrams, open(os.path.join(output_folder, f"{args.prefix}_right_hold_diagrams.pkl"), "wb"))
    pickle.dump(right_resting_diagrams, open(os.path.join(output_folder, f"{args.prefix}_right_resting_diagrams.pkl"), "wb"))

    # ========================================================================
    # STEP 7: Feature Extraction
    # ========================================================================
    print(f"\n[7/8] Extracting topological features...")

    all_features = {}

    # 7.1 Persistence Entropy
    print("\n  Computing Persistence Entropy...")
    PE = PersistenceEntropy()
    all_features['left_hold_pe'] = [PE.fit_transform(d) for d in left_hold_diagrams]
    all_features['left_resting_pe'] = [PE.fit_transform(d) for d in left_resting_diagrams]
    all_features['right_hold_pe'] = [PE.fit_transform(d) for d in right_hold_diagrams]
    all_features['right_resting_pe'] = [PE.fit_transform(d) for d in right_resting_diagrams]

    # 7.2 Summary Statistics Features
    print("  Computing summary statistics features...")
    all_features['left_hold_stats'] = [extract_features(d, homology_dimensions=homology_dims, verbose=False) for d in left_hold_diagrams]
    all_features['left_resting_stats'] = [extract_features(d, homology_dimensions=homology_dims, verbose=False) for d in left_resting_diagrams]
    all_features['right_hold_stats'] = [extract_features(d, homology_dimensions=homology_dims, verbose=False) for d in right_hold_diagrams]
    all_features['right_resting_stats'] = [extract_features(d, homology_dimensions=homology_dims, verbose=False) for d in right_resting_diagrams]

    # 7.3 Persistence Landscapes
    print("  Computing Persistence Landscapes...")
    PL = PersistenceLandscape()
    all_features['left_hold_pl'] = [PL.fit_transform(d) for d in left_hold_diagrams]
    all_features['left_resting_pl'] = [PL.fit_transform(d) for d in left_resting_diagrams]
    all_features['right_hold_pl'] = [PL.fit_transform(d) for d in right_hold_diagrams]
    all_features['right_resting_pl'] = [PL.fit_transform(d) for d in right_resting_diagrams]

    # 7.4 Betti Curves
    print("  Computing Betti Curves...")
    BC = BettiCurve()
    all_features['left_hold_bc'] = [BC.fit_transform(d) for d in left_hold_diagrams]
    all_features['left_resting_bc'] = [BC.fit_transform(d) for d in left_resting_diagrams]
    all_features['right_hold_bc'] = [BC.fit_transform(d) for d in right_hold_diagrams]
    all_features['right_resting_bc'] = [BC.fit_transform(d) for d in right_resting_diagrams]

    # 7.5 Heat Kernel
    print("  Computing Heat Kernel signatures...")
    HK = HeatKernel()
    all_features['left_hold_hk'] = [HK.fit_transform(d) for d in left_hold_diagrams]
    all_features['left_resting_hk'] = [HK.fit_transform(d) for d in left_resting_diagrams]
    all_features['right_hold_hk'] = [HK.fit_transform(d) for d in right_hold_diagrams]
    all_features['right_resting_hk'] = [HK.fit_transform(d) for d in right_resting_diagrams]

    # ========================================================================
    # STEP 8: Compute Distance Matrices (if requested)
    # ========================================================================
    if args.compute_distances:
        print(f"\n[8/8] Computing pairwise distance matrices...")

        # Function to compute distances for a set of diagrams
        def compute_distance_matrix(diagrams, scaler_metric, distance_metric, name):
            print(f"\n  {name} ({distance_metric} distance with {scaler_metric} scaling)...")

            # Extract and pad diagrams
            diagrams_2d = [d[0] for d in diagrams]
            diagrams_padded = pad_diagrams(diagrams_2d)

            # Scale
            scaler = Scaler(metric=scaler_metric)
            diagrams_scaled = scaler.fit_transform(diagrams_padded)

            # Compute distances
            pwise_dist = PairwiseDistance(metric=distance_metric)
            dist_matrix = pwise_dist.fit_transform(diagrams_scaled)

            return dist_matrix

        # Combine hold and resting for each channel for distance calculation
        left_all_diagrams = left_hold_diagrams + left_resting_diagrams
        right_all_diagrams = right_hold_diagrams + right_resting_diagrams

        # Wasserstein distances
        all_features['left_wasserstein'] = compute_distance_matrix(
            left_all_diagrams, 'wasserstein', 'wasserstein', 'Left Channel Wasserstein'
        )
        all_features['right_wasserstein'] = compute_distance_matrix(
            right_all_diagrams, 'wasserstein', 'wasserstein', 'Right Channel Wasserstein'
        )

        # Bottleneck distances
        all_features['left_bottleneck'] = compute_distance_matrix(
            left_all_diagrams, 'wasserstein', 'bottleneck', 'Left Channel Bottleneck'
        )
        all_features['right_bottleneck'] = compute_distance_matrix(
            right_all_diagrams, 'wasserstein', 'bottleneck', 'Right Channel Bottleneck'
        )
    else:
        print(f"\n[8/8] Skipping distance matrix computation (use --compute-distances to enable)")

    # ========================================================================
    # Save All Features
    # ========================================================================
    print(f"\nSaving all features to {output_folder}...")
    features_file = os.path.join(output_folder, f"{args.prefix}_all_features.pkl")
    with open(features_file, "wb") as f:
        pickle.dump(all_features, f)

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nOutput saved to: {output_folder}")
    print(f"  - Persistence diagrams: *_diagrams.pkl")
    print(f"  - All features: {args.prefix}_all_features.pkl")
    if args.save_timeseries:
        print(f"  - Time series slices: *_hold.pkl, *_resting.pkl")

    # Print feature summary
    print(f"\nFeature Summary:")
    print(f"  - Persistence Entropy: {len(all_features['left_hold_pe'])} left hold, {len(all_features['left_resting_pe'])} left resting")
    print(f"  - Summary Statistics: Available for all slices")
    print(f"  - Persistence Landscapes: Shape {all_features['left_hold_pl'][0].shape}")
    print(f"  - Betti Curves: Available for all slices")
    print(f"  - Heat Kernel: Available for all slices")
    if args.compute_distances:
        print(f"  - Distance Matrices: Wasserstein and Bottleneck computed")


def main():
    parser = argparse.ArgumentParser(
        description='Extract topological features from Parkinson\'s LFP data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_extraction.py data.mat events.txt ./output
  python feature_extraction.py data.mat events.txt ./output --prefix medOff --compute-distances
  python feature_extraction.py data.mat events.txt ./output --lowcut 8 --highcut 30
        """
    )

    # Required arguments
    parser.add_argument('mat_file', help='Path to .mat data file')
    parser.add_argument('event_times_file', help='Path to event times text file')
    parser.add_argument('output_folder', help='Directory to save output files')

    # Optional arguments
    parser.add_argument('--prefix', default='data',
                        help='Prefix for output files (default: data)')
    parser.add_argument('--lowcut', type=float, default=4.0,
                        help='Lower frequency for bandpass filter in Hz (default: 4)')
    parser.add_argument('--highcut', type=float, default=48.0,
                        help='Upper frequency for bandpass filter in Hz (default: 48)')
    parser.add_argument('--target-fs', type=int, default=100,
                        help='Target sampling frequency after downsampling in Hz (default: 100)')
    parser.add_argument('--slice-length', type=float, default=60.0,
                        help='Length of each time slice in seconds (default: 60)')
    parser.add_argument('--max-embedding-dim', type=int, default=20,
                        help='Maximum dimension for Takens embedding search (default: 20)')
    parser.add_argument('--max-time-delay', type=int, default=1000,
                        help='Maximum time delay for Takens embedding search (default: 1000)')
    parser.add_argument('--embedding-subset-size', type=int, default=250,
                        help='Number of points to use for persistence calculation (default: 250)')
    parser.add_argument('--save-timeseries', action='store_true',
                        help='Save sliced time series data')
    parser.add_argument('--compute-distances', action='store_true',
                        help='Compute pairwise distance matrices (Wasserstein and Bottleneck)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.mat_file):
        print(f"ERROR: Data file not found: {args.mat_file}")
        sys.exit(1)

    if not os.path.exists(args.event_times_file):
        print(f"ERROR: Event times file not found: {args.event_times_file}")
        sys.exit(1)

    # Run the pipeline
    process_pipeline(args.mat_file, args.event_times_file, args.output_folder, args)


if __name__ == "__main__":
    main()
