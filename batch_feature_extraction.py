#!/usr/bin/env python3
"""
Batch Feature Extraction Script for Parkinson's LFP Data
Simplified version that processes all patients at once with single slices.

This script:
- Processes all 14 patients automatically
- Extracts ONE 5-second slice from resting period
- Extracts ONE 5-second slice from first hold event
- Avoids edge artifacts by centering slices
- Handles patients with only medOn or only medOff data
- Generates plots for all topological features:
  * Persistence diagrams
  * Betti curves (H0-H3)
  * Persistence landscapes
  * Heat kernels
- Saves features and plots in patient directories following naming conventions
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
from scipy.signal import decimate
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for batch processing
matplotlib.use('Agg')

# Import custom utilities
from eeg_utils import (
    mat_to_dataframe,
    butter_bandpass_filter,
    fit_embedder,
    extract_features,
)

# Import TDA libraries
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (
    PersistenceEntropy,
    PersistenceLandscape,
    BettiCurve,
    HeatKernel,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directory
DATA_DIR = "./sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert"

# Patient IDs (with event times extracted)
PATIENT_IDS = [
    '0cGdk9', '2IhVOz', '2IU8mi', 'AB2PeX', 'AbzsOg', 'BYJoWR',
    'dCsWjQ', 'FYbcap', 'gNX5yb', 'i4oK0F', 'jyC0j3', 'PuPVlx',
    'QZTsn6', 'VopvKx'
]

# Processing parameters
LOWCUT = 4.0  # Hz
HIGHCUT = 48.0  # Hz
ORIGINAL_FS = 2000  # Hz
TARGET_FS = 100  # Hz
SLICE_LENGTH = 5.0  # seconds (changed from 60 to 5)
MAX_EMBEDDING_DIM = 10  # Reduced from 20 for 5-second slices
MAX_TIME_DELAY = 50  # Reduced from 1000 to fit 5-second slices (500 samples at 100 Hz)
EMBEDDING_SUBSET_SIZE = 250
HOMOLOGY_DIMS = [0, 1, 2, 3]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_hold_type(filename):
    """Detect whether data is HoldL or HoldR from filename."""
    if 'HoldL' in filename:
        return 'holdL'
    elif 'HoldR' in filename:
        return 'holdR'
    return None


def map_to_dominant(channel_name, hold_type):
    """
    Map hemisphere name to dominant/nondominant based on hold type.

    Based on contralateral motor control:
    - holdL (left arm raised) -> right hemisphere is dominant
    - holdR (right arm raised) -> left hemisphere is dominant

    Args:
        channel_name: 'left' or 'right'
        hold_type: 'holdL' or 'holdR'

    Returns:
        'dominant' or 'nondominant'
    """
    if hold_type == 'holdL':
        # Left arm raised -> right hemisphere is dominant
        return 'dominant' if channel_name == 'right' else 'nondominant'
    elif hold_type == 'holdR':
        # Right arm raised -> left hemisphere is dominant
        return 'dominant' if channel_name == 'left' else 'nondominant'
    else:
        # If hold type unknown, keep original
        return channel_name


def load_event_times(event_file, med_state):
    """
    Load event times for a specific medication state.

    Returns:
        dict: {'hold': [event_dicts], 'resting': [event_dicts]}
    """
    print(f"  Loading event times for {med_state}...")

    events = {'hold': [], 'resting': []}
    current_section = None
    in_correct_section = False

    with open(event_file, 'r') as f:
        lines = f.readlines()

    # Count sections to handle single-section files
    section_count = sum(1 for line in lines if line.strip().startswith('---') and 'events.tsv' in line.lower())
    use_single_section = (section_count <= 1)

    for line in lines:
        line = line.strip()

        # Check for section headers
        if line.startswith('---') and 'events.tsv' in line.lower():
            if 'medon' in line.lower():
                current_section = 'MedOn'
            elif 'medoff' in line.lower():
                current_section = 'MedOff'

            # Determine if we should use this section
            if use_single_section or med_state.lower() in current_section.lower():
                in_correct_section = True
            else:
                in_correct_section = False
            continue

        # Skip headers and empty lines
        if not line or line.startswith('#') or line.startswith('trial_type'):
            continue

        if not in_correct_section:
            continue

        # Parse event data
        parts = line.split()
        if len(parts) >= 4:
            try:
                trial_type = parts[0]
                onset = float(parts[1])
                duration = float(parts[2])
                end = float(parts[3])

                event_info = {
                    'onset': onset,
                    'duration': duration,
                    'end': end
                }

                if 'hold' in trial_type.lower():
                    events['hold'].append(event_info)
                elif 'rest' in trial_type.lower():
                    events['resting'].append(event_info)
            except (ValueError, IndexError):
                pass

    print(f"    Found {len(events['hold'])} hold events, {len(events['resting'])} resting events")
    return events


def extract_centered_slice(signal, event, fs, slice_length, avoid_edge_buffer=1.0):
    """
    Extract a centered slice from an event, avoiding edges.

    Args:
        signal: The full signal array
        event: Event dict with 'onset', 'end', 'duration'
        fs: Sampling frequency
        slice_length: Length of slice in seconds
        avoid_edge_buffer: Minimum distance from event edges in seconds

    Returns:
        np.ndarray: The extracted slice, or None if slice doesn't fit
    """
    if event['end'] is None or event['onset'] is None:
        return None

    event_duration = event['end'] - event['onset']

    # Check if event is long enough for slice + buffers
    min_required_duration = slice_length + 2 * avoid_edge_buffer
    if event_duration < min_required_duration:
        print(f"    Warning: Event too short ({event_duration:.2f}s) for {slice_length}s slice with {avoid_edge_buffer}s buffers")
        # Try without buffer
        if event_duration >= slice_length:
            avoid_edge_buffer = 0
        else:
            return None

    # Calculate centered position
    event_center = event['onset'] + event_duration / 2.0
    slice_start = event_center - slice_length / 2.0
    slice_end = event_center + slice_length / 2.0

    # Ensure we stay within the allowed region (event boundaries minus buffers)
    min_allowed_start = event['onset'] + avoid_edge_buffer
    max_allowed_end = event['end'] - avoid_edge_buffer

    # Adjust if needed
    if slice_start < min_allowed_start:
        slice_start = min_allowed_start
        slice_end = slice_start + slice_length
    elif slice_end > max_allowed_end:
        slice_end = max_allowed_end
        slice_start = slice_end - slice_length

    # Convert to sample indices
    start_idx = int(slice_start * fs)
    end_idx = int(slice_end * fs)

    # Bounds check
    if start_idx < 0 or end_idx > len(signal):
        return None

    slice_data = signal[start_idx:end_idx]
    print(f"    Extracted slice: [{slice_start:.2f}s - {slice_end:.2f}s] from event [{event['onset']:.2f}s - {event['end']:.2f}s]")

    return slice_data


def plot_persistence_diagram(diagram, output_path, title="Persistence Diagram"):
    """
    Plot and save a persistence diagram.

    Args:
        diagram: Persistence diagram array with shape (n_samples, n_features, 3)
                 where columns are [birth, death, dimension]
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Extract the first sample (we have shape (1, n_features, 3))
    diagram_2d = diagram[0]

    # Separate by homology dimension
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3'}

    fig, ax = plt.subplots(figsize=(8, 8))

    # Find max value for diagonal line (exclude infinite values)
    finite_mask = np.isfinite(diagram_2d[:, 1])
    if np.any(finite_mask):
        max_val = max(
            np.max(diagram_2d[finite_mask, 0]),
            np.max(diagram_2d[finite_mask, 1])
        )
    else:
        max_val = 1.0

    # Plot diagonal line (birth = death)
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, label='Birth = Death')

    # Plot points for each homology dimension
    for dim in HOMOLOGY_DIMS:
        mask = diagram_2d[:, 2] == dim
        if np.any(mask):
            births = diagram_2d[mask, 0]
            deaths = diagram_2d[mask, 1]

            # Separate finite and infinite deaths
            finite_deaths = np.isfinite(deaths)

            if np.any(finite_deaths):
                ax.scatter(
                    births[finite_deaths],
                    deaths[finite_deaths],
                    c=colors[dim],
                    label=f'{labels[dim]} ({np.sum(finite_deaths)} features)',
                    alpha=0.6,
                    s=50
                )

            # Mark infinite deaths with triangles at top of plot
            if np.any(~finite_deaths):
                ax.scatter(
                    births[~finite_deaths],
                    [max_val * 1.05] * np.sum(~finite_deaths),
                    c=colors[dim],
                    marker='^',
                    s=100,
                    alpha=0.6
                )

    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_betti_curves(betti_curve, output_path, title="Betti Curves"):
    """
    Plot and save Betti curves for all homology dimensions.

    Args:
        betti_curve: Betti curve array with shape (n_samples, n_bins, n_homology_dims)
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Extract the first sample
    betti_2d = betti_curve[0]  # Shape: (n_bins, n_homology_dims)

    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3'}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each homology dimension
    for i, dim in enumerate(HOMOLOGY_DIMS):
        if i < betti_2d.shape[1]:
            ax.plot(betti_2d[:, i], color=colors[dim], label=labels[dim], linewidth=2, alpha=0.7)

    ax.set_xlabel('Filtration Value', fontsize=12)
    ax.set_ylabel('Betti Number', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_persistence_landscape(landscape, output_path, title="Persistence Landscape"):
    """
    Plot and save persistence landscape as a heatmap.

    Args:
        landscape: Persistence landscape array with shape (n_samples, n_layers, n_bins)
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Extract the first sample
    landscape_2d = landscape[0]  # Shape: (n_layers, n_bins)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot as heatmap
    im = ax.imshow(landscape_2d, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Bin Index', fontsize=12)
    ax.set_ylabel('Landscape Layer', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heat_kernel(heat_kernel, output_path, title="Heat Kernel"):
    """
    Plot and save heat kernel signature.

    Args:
        heat_kernel: Heat kernel array with shape (n_samples, ...)
                     Could be (n_samples, n_bins) or (n_samples, n_homology_dims, n_bins, n_bins)
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Extract the first sample
    heat_data = heat_kernel[0]

    # Check dimensions and handle accordingly
    if heat_data.ndim == 1:
        # Simple 1D case: (n_bins,)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(heat_data, color='darkblue', linewidth=2, alpha=0.7)
        ax.fill_between(range(len(heat_data)), heat_data, alpha=0.3, color='lightblue')
        ax.set_xlabel('Bin Index', fontsize=12)
        ax.set_ylabel('Heat Kernel Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    elif heat_data.ndim == 2:
        # 2D case: plot as heatmap (n_bins, n_bins) or as lines (n_homology_dims, n_bins)
        if heat_data.shape[0] <= 4:  # Likely homology dimensions
            # Plot each dimension as a separate line
            colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
            labels = {0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3'}

            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(heat_data.shape[0]):
                if i < len(colors):
                    ax.plot(heat_data[i], color=colors[i], label=labels[i], linewidth=2, alpha=0.7)
            ax.set_xlabel('Bin Index', fontsize=12)
            ax.set_ylabel('Heat Kernel Value', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            # Plot as heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(heat_data, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_xlabel('Bin Index', fontsize=12)
            ax.set_ylabel('Bin Index', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Heat Kernel Value', fontsize=10)

    elif heat_data.ndim == 3:
        # 3D case: (n_homology_dims, n_bins, n_bins) - flatten by summing across bins
        # and plot each homology dimension
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
        labels = {0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3'}

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(heat_data.shape[0]):
            # Sum across one dimension to get a 1D representation
            heat_1d = np.sum(heat_data[i], axis=0)
            if i < len(colors):
                ax.plot(heat_1d, color=colors[i], label=labels[i], linewidth=2, alpha=0.7)
        ax.set_xlabel('Bin Index', fontsize=12)
        ax.set_ylabel('Summed Heat Kernel Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    else:
        # Fallback: flatten everything and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        heat_flat = heat_data.flatten()
        ax.plot(heat_flat, color='darkblue', linewidth=1, alpha=0.7)
        ax.set_xlabel('Flattened Index', fontsize=12)
        ax.set_ylabel('Heat Kernel Value', fontsize=12)
        ax.set_title(f"{title} (Flattened)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_patient(patient_id):
    """Process a single patient's data."""

    print(f"\n{'='*80}")
    print(f"PROCESSING PATIENT: {patient_id}")
    print(f"{'='*80}")

    # Find patient directory
    patient_dir = os.path.join('.', patient_id)
    if not os.path.exists(patient_dir):
        print(f"  ERROR: Patient directory not found: {patient_dir}")
        return False

    # Load event times
    event_file = os.path.join(patient_dir, 'event_times.txt')
    if not os.path.exists(event_file):
        print(f"  ERROR: Event times file not found: {event_file}")
        return False

    # Find .mat files for this patient
    mat_pattern = os.path.join(DATA_DIR, f"*{patient_id}*.mat")
    mat_files = glob.glob(mat_pattern)

    if not mat_files:
        print(f"  ERROR: No .mat files found for patient {patient_id}")
        return False

    print(f"  Found {len(mat_files)} .mat file(s)")

    # Process each .mat file (medOn and/or medOff)
    for mat_file in mat_files:
        basename = os.path.basename(mat_file)
        print(f"\n  Processing file: {basename}")

        # Determine medication state
        if 'MedOn' in basename:
            med_state = 'medOn'
        elif 'MedOff' in basename:
            med_state = 'medOff'
        else:
            print(f"    WARNING: Could not determine medication state from filename")
            continue

        # Detect hold type
        hold_type = detect_hold_type(basename)
        if not hold_type:
            print(f"    WARNING: Could not detect hold type")
            hold_suffix = ""
        else:
            hold_suffix = f"_{hold_type}"

        print(f"    Medication state: {med_state}, Hold type: {hold_type}")

        # Process this file
        success = process_single_file(
            mat_file=mat_file,
            event_file=event_file,
            output_dir=patient_dir,
            med_state=med_state,
            hold_suffix=hold_suffix
        )

        if not success:
            print(f"    Failed to process {basename}")
            continue

    print(f"\n  ✓ Completed patient {patient_id}")
    return True


def process_single_file(mat_file, event_file, output_dir, med_state, hold_suffix):
    """Process a single .mat file and extract features."""

    try:
        # ====================================================================
        # STEP 1: Load Data
        # ====================================================================
        print(f"\n    [1/7] Loading LFP data...")
        result = mat_to_dataframe(mat_file)
        if result is None:
            print("      ERROR: Failed to load data")
            return False

        df, left_lfp, right_lfp, left_name, right_name = result

        # ====================================================================
        # STEP 2: Bandpass Filtering
        # ====================================================================
        print(f"    [2/7] Applying bandpass filter ({LOWCUT}-{HIGHCUT} Hz)...")
        left_filtered = butter_bandpass_filter(left_lfp, LOWCUT, HIGHCUT, ORIGINAL_FS, order=5)
        right_filtered = butter_bandpass_filter(right_lfp, LOWCUT, HIGHCUT, ORIGINAL_FS, order=5)
        print(f"      ✓ Filtered {left_name} and {right_name}")

        # ====================================================================
        # STEP 3: Downsampling
        # ====================================================================
        print(f"    [3/7] Downsampling from {ORIGINAL_FS} Hz to {TARGET_FS} Hz...")
        q = ORIGINAL_FS // TARGET_FS
        left_downsampled = decimate(x=left_filtered, q=q, ftype="fir", zero_phase=True)
        right_downsampled = decimate(x=right_filtered, q=q, ftype="fir", zero_phase=True)
        print(f"      ✓ Downsampled: {len(left_filtered)} → {len(left_downsampled)} samples")

        # ====================================================================
        # STEP 4: Load Events and Extract Single Slices
        # ====================================================================
        print(f"    [4/7] Loading events and extracting single slices...")
        events = load_event_times(event_file, med_state)

        # Extract ONE slice from resting (from first resting event)
        left_resting_slice = None
        right_resting_slice = None
        if events['resting']:
            print(f"      Extracting resting slice from first event...")
            left_resting_slice = extract_centered_slice(
                left_downsampled, events['resting'][0], TARGET_FS, SLICE_LENGTH
            )
            right_resting_slice = extract_centered_slice(
                right_downsampled, events['resting'][0], TARGET_FS, SLICE_LENGTH
            )
        else:
            print(f"      WARNING: No resting events found")

        # Extract ONE slice from hold (from first hold event)
        left_hold_slice = None
        right_hold_slice = None
        if events['hold']:
            print(f"      Extracting hold slice from first event...")
            left_hold_slice = extract_centered_slice(
                left_downsampled, events['hold'][0], TARGET_FS, SLICE_LENGTH
            )
            right_hold_slice = extract_centered_slice(
                right_downsampled, events['hold'][0], TARGET_FS, SLICE_LENGTH
            )
        else:
            print(f"      WARNING: No hold events found")

        # Collect valid slices
        slices_to_process = {}

        if left_hold_slice is not None and right_hold_slice is not None:
            slices_to_process['hold'] = {
                'left': left_hold_slice,
                'right': right_hold_slice
            }

        if left_resting_slice is not None and right_resting_slice is not None:
            slices_to_process['resting'] = {
                'left': left_resting_slice,
                'right': right_resting_slice
            }

        if not slices_to_process:
            print(f"      ERROR: No valid slices extracted")
            return False

        print(f"      ✓ Extracted {len(slices_to_process)} condition(s)")

        # ====================================================================
        # STEP 5: Takens Embedding
        # ====================================================================
        print(f"    [5/7] Computing Takens embeddings...")

        # Validate and clean data before embedding
        print(f"      Validating data...")
        for condition, channels in slices_to_process.items():
            for channel_name, slice_data in channels.items():
                # Check for NaN or inf
                if np.any(np.isnan(slice_data)):
                    print(f"        WARNING: NaN values found in {channel_name} {condition}, replacing with 0")
                    slice_data = np.nan_to_num(slice_data, nan=0.0)
                    slices_to_process[condition][channel_name] = slice_data

                if np.any(np.isinf(slice_data)):
                    print(f"        WARNING: Inf values found in {channel_name} {condition}, replacing with 0")
                    slice_data = np.nan_to_num(slice_data, posinf=0.0, neginf=0.0)
                    slices_to_process[condition][channel_name] = slice_data

                # Ensure data is not constant (would cause issues in MI calculation)
                if np.std(slice_data) == 0:
                    print(f"        WARNING: Constant signal in {channel_name} {condition}")
                    # Add tiny noise to avoid constant signal
                    slice_data = slice_data + np.random.normal(0, 1e-10, len(slice_data))
                    slices_to_process[condition][channel_name] = slice_data

        embedder = SingleTakensEmbedding(
            parameters_type="search",
            time_delay=MAX_TIME_DELAY,
            dimension=MAX_EMBEDDING_DIM,
            stride=1,
            n_jobs=-1
        )

        embeddings = {}
        for condition, channels in slices_to_process.items():
            embeddings[condition] = {}
            for channel_name, slice_data in channels.items():
                print(f"      Embedding {channel_name} {condition}...", end=" ")
                try:
                    emb = fit_embedder(embedder, slice_data, verbose=False)
                    embeddings[condition][channel_name] = emb
                    print(f"✓ Shape: {emb.shape}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(f"        Data stats: mean={np.mean(slice_data):.3f}, std={np.std(slice_data):.3f}, min={np.min(slice_data):.3f}, max={np.max(slice_data):.3f}")
                    raise

        # ====================================================================
        # STEP 6: Persistent Homology
        # ====================================================================
        print(f"    [6/7] Computing persistent homology...")
        persistence = VietorisRipsPersistence(homology_dimensions=HOMOLOGY_DIMS, n_jobs=-1)

        diagrams = {}
        for condition, channels in embeddings.items():
            diagrams[condition] = {}
            for channel_name, embedding in channels.items():
                print(f"      Computing diagram for {channel_name} {condition}...", end=" ")
                embedding_subset = embedding[:EMBEDDING_SUBSET_SIZE]
                embedding_3d = embedding_subset[None, :, :]
                diagram = persistence.fit_transform(embedding_3d)
                diagrams[condition][channel_name] = diagram
                print("✓")

        # Determine hold type for dominant mapping
        hold_type = hold_suffix.strip('_') if hold_suffix else None

        # Save persistence diagrams and generate plots
        for condition in diagrams.keys():
            for channel_name in diagrams[condition].keys():
                # Map channel name to dominant/nondominant
                mapped_name = map_to_dominant(channel_name, hold_type) if hold_type else channel_name

                # Naming: {medState}_{dominant/nondominant}_{condition}{holdSuffix}_diagrams.pkl
                if condition == 'hold':
                    base_filename = f"{med_state}_{mapped_name}_{condition}{hold_suffix}"
                else:
                    base_filename = f"{med_state}_{mapped_name}_{condition}"

                # Save pickle file
                pkl_filename = f"{base_filename}_diagrams.pkl"
                pkl_filepath = os.path.join(output_dir, pkl_filename)
                with open(pkl_filepath, 'wb') as f:
                    pickle.dump(diagrams[condition][channel_name], f)

                # Generate and save plot
                plot_filename = f"{base_filename}_diagram.png"
                plot_filepath = os.path.join(output_dir, plot_filename)
                plot_title = f"{med_state.upper()} - {mapped_name.capitalize()} - {condition.capitalize()}"
                if condition == 'hold' and hold_suffix:
                    plot_title += f" ({hold_suffix.strip('_').upper()})"

                print(f"      Saving plot: {plot_filename}", end=" ")
                plot_persistence_diagram(
                    diagrams[condition][channel_name],
                    plot_filepath,
                    title=plot_title
                )
                print("✓")

        # ====================================================================
        # STEP 7: Feature Extraction and Plotting
        # ====================================================================
        print(f"    [7/7] Extracting topological features and generating plots...")

        all_features = {}

        for condition in diagrams.keys():
            all_features[condition] = {}

            for channel_name in diagrams[condition].keys():
                diagram = diagrams[condition][channel_name]

                # Map channel name to dominant/nondominant
                mapped_name = map_to_dominant(channel_name, hold_type) if hold_type else channel_name

                # Store features under mapped name
                all_features[condition][mapped_name] = {}

                # Determine base filename for this condition/channel (using mapped name)
                if condition == 'hold':
                    base_filename = f"{med_state}_{mapped_name}_{condition}{hold_suffix}"
                else:
                    base_filename = f"{med_state}_{mapped_name}_{condition}"

                # Create plot title base
                plot_title_base = f"{med_state.upper()} - {mapped_name.capitalize()} - {condition.capitalize()}"
                if condition == 'hold' and hold_suffix:
                    plot_title_base += f" ({hold_suffix.strip('_').upper()})"

                # 7.1 Persistence Entropy
                PE = PersistenceEntropy()
                all_features[condition][mapped_name]['persistence_entropy'] = PE.fit_transform(diagram)

                # 7.2 Summary Statistics
                all_features[condition][mapped_name]['stats'] = extract_features(
                    diagram, homology_dimensions=HOMOLOGY_DIMS, verbose=False
                )

                # 7.3 Persistence Landscape
                PL = PersistenceLandscape()
                landscape = PL.fit_transform(diagram)
                all_features[condition][mapped_name]['persistence_landscape'] = landscape

                # Plot persistence landscape
                landscape_plot_path = os.path.join(output_dir, f"{base_filename}_landscape.png")
                plot_persistence_landscape(landscape, landscape_plot_path, title=f"{plot_title_base} - Landscape")

                # 7.4 Betti Curve
                BC = BettiCurve()
                betti_curve = BC.fit_transform(diagram)
                all_features[condition][mapped_name]['betti_curve'] = betti_curve

                # Plot Betti curves
                betti_plot_path = os.path.join(output_dir, f"{base_filename}_betti.png")
                plot_betti_curves(betti_curve, betti_plot_path, title=f"{plot_title_base} - Betti Curves")

                # 7.5 Heat Kernel
                HK = HeatKernel()
                heat_kernel = HK.fit_transform(diagram)
                all_features[condition][mapped_name]['heat_kernel'] = heat_kernel

                # Plot heat kernel
                heat_plot_path = os.path.join(output_dir, f"{base_filename}_heat.png")
                plot_heat_kernel(heat_kernel, heat_plot_path, title=f"{plot_title_base} - Heat Kernel")

        print(f"      ✓ Extracted all features and generated plots")

        # Save all features
        # Naming: {medState}_all_features{holdSuffix}.pkl
        features_file = os.path.join(output_dir, f"{med_state}_all_features{hold_suffix}.pkl")
        with open(features_file, 'wb') as f:
            pickle.dump(all_features, f)

        print(f"      ✓ Saved to {features_file}")

        return True

    except Exception as e:
        print(f"      ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Batch feature extraction for Parkinson\'s LFP data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients
  python batch_feature_extraction.py

  # Process a specific patient
  python batch_feature_extraction.py --patient FYbcap

  # Process multiple specific patients
  python batch_feature_extraction.py --patient i4oK0F QZTsn6 AB2PeX
        """
    )
    parser.add_argument(
        '--patient', '--patients',
        nargs='+',
        metavar='PATIENT_ID',
        help='Process specific patient(s) only. Provide one or more patient IDs.'
    )

    args = parser.parse_args()

    # Determine which patients to process
    if args.patient:
        # Validate patient IDs
        invalid_patients = [p for p in args.patient if p not in PATIENT_IDS]
        if invalid_patients:
            print(f"ERROR: Invalid patient ID(s): {', '.join(invalid_patients)}")
            print(f"Valid patient IDs: {', '.join(PATIENT_IDS)}")
            sys.exit(1)

        patients_to_process = args.patient
        print("="*80)
        print("BATCH FEATURE EXTRACTION FOR PARKINSON'S LFP DATA")
        print("="*80)
        print(f"\nProcessing {len(patients_to_process)} specific patient(s): {', '.join(patients_to_process)}")
    else:
        patients_to_process = PATIENT_IDS
        print("="*80)
        print("BATCH FEATURE EXTRACTION FOR PARKINSON'S LFP DATA")
        print("="*80)
        print(f"\nProcessing {len(patients_to_process)} patients with simplified single-slice extraction")

    print(f"Slice length: {SLICE_LENGTH} seconds")
    print(f"Taking 1 resting slice and 1 hold slice per medication state\n")

    # Process each patient
    success_count = 0
    failed_patients = []

    for patient_id in patients_to_process:
        success = process_patient(patient_id)
        if success:
            success_count += 1
        else:
            failed_patients.append(patient_id)

    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(patients_to_process)} patients")

    if failed_patients:
        print(f"\nFailed patients: {', '.join(failed_patients)}")
    else:
        print("\n✓ All patients processed successfully!")


if __name__ == "__main__":
    main()
