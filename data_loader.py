"""
Data Loader Module for Parkinson's Disease TDA Analysis

This module provides utilities for loading and organizing topological features
extracted from LFP recordings of Parkinson's disease patients in medOn and medOff states.

Features are organized by:
- patient_id: 14 patients total
- med_state: 'medOn' or 'medOff'
- hemisphere: 'dominant' or 'nondominant' (based on contralateral motor control)
- condition: 'hold' (motor task) or 'resting'

Usage:
    from data_loader import load_all_patients, load_patient_features

    # Load all patients with verbose output
    df = load_all_patients(verbose=True)

    # Load specific patient
    features = load_patient_features('i4oK0F', verbose=True)
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Base directory (assume script is in project root)
BASE_DIR = Path(__file__).parent.resolve()

# Patient lists by data availability
PATIENTS_ALL = [
    '0cGdk9', '2IhVOz', '2IU8mi', 'AB2PeX', 'AbzsOg', 'BYJoWR',
    'dCsWjQ', 'FYbcap', 'gNX5yb', 'i4oK0F', 'jyC0j3', 'PuPVlx',
    'QZTsn6', 'VopvKx'
]

# Patients with both medOn and medOff (n=9, most useful for paired comparisons)
PATIENTS_PAIRED = [
    '0cGdk9', '2IhVOz', '2IU8mi', 'BYJoWR', 'QZTsn6',
    'dCsWjQ', 'i4oK0F', 'jyC0j3', 'AbzsOg'
]

# Patients with only medOff (n=3)
PATIENTS_MEDOFF_ONLY = ['AB2PeX', 'gNX5yb', 'PuPVlx']

# Patients with only medOn (n=2)
PATIENTS_MEDON_ONLY = ['FYbcap', 'VopvKx']

# Hold type by patient (for file naming)
PATIENT_HOLD_TYPE = {
    '0cGdk9': 'holdL', '2IU8mi': 'holdL', 'AB2PeX': 'holdL', 'AbzsOg': 'holdL',
    'FYbcap': 'holdL', 'PuPVlx': 'holdL', 'QZTsn6': 'holdL', 'dCsWjQ': 'holdL',
    'gNX5yb': 'holdL', 'i4oK0F': 'holdL',
    '2IhVOz': 'holdR', 'BYJoWR': 'holdR', 'VopvKx': 'holdR', 'jyC0j3': 'holdR'
}


def get_patient_availability() -> Dict[str, Dict]:
    """
    Get information about which medication states are available for each patient.

    Returns:
        Dictionary mapping patient_id to availability info:
        {
            'patient_id': {
                'has_medOn': bool,
                'has_medOff': bool,
                'hold_type': 'holdL' or 'holdR',
                'states': list of available states
            }
        }
    """
    availability = {}

    for patient in PATIENTS_ALL:
        info = {
            'has_medOn': patient in PATIENTS_PAIRED or patient in PATIENTS_MEDON_ONLY,
            'has_medOff': patient in PATIENTS_PAIRED or patient in PATIENTS_MEDOFF_ONLY,
            'hold_type': PATIENT_HOLD_TYPE[patient]
        }
        info['states'] = []
        if info['has_medOn']:
            info['states'].append('medOn')
        if info['has_medOff']:
            info['states'].append('medOff')

        availability[patient] = info

    return availability


def load_patient_features(patient_id: str, verbose: bool = False) -> Dict:
    """
    Load all available features for a single patient.

    Args:
        patient_id: Patient identifier (e.g., 'i4oK0F')
        verbose: If True, print detailed loading information

    Returns:
        Dictionary with structure:
        {
            'patient_id': str,
            'hold_type': 'holdL' or 'holdR',
            'medOn': {features_dict} or None,
            'medOff': {features_dict} or None
        }

        Each features_dict has structure:
        {
            'hold': {
                'dominant': {feature_data},
                'nondominant': {feature_data}
            },
            'resting': {
                'dominant': {feature_data},
                'nondominant': {feature_data}
            }
        }
    """
    if patient_id not in PATIENTS_ALL:
        raise ValueError(f"Unknown patient: {patient_id}. Must be one of {PATIENTS_ALL}")

    patient_dir = BASE_DIR / patient_id
    hold_type = PATIENT_HOLD_TYPE[patient_id]
    availability = get_patient_availability()[patient_id]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading patient: {patient_id}")
        print(f"Hold type: {hold_type}")
        print(f"Available states: {', '.join(availability['states'])}")
        print(f"{'='*70}")

    result = {
        'patient_id': patient_id,
        'hold_type': hold_type,
        'medOn': None,
        'medOff': None
    }

    # Load each available medication state
    for med_state in availability['states']:
        feature_file = patient_dir / f"{med_state}_all_features_{hold_type}.pkl"

        if not feature_file.exists():
            if verbose:
                print(f"  ⚠ Warning: Expected file not found: {feature_file}")
            continue

        try:
            with open(feature_file, 'rb') as f:
                features = pickle.load(f)

            result[med_state] = features

            if verbose:
                print(f"\n  ✓ Loaded {med_state} features:")
                print(f"    Conditions: {list(features.keys())}")
                for condition in features.keys():
                    print(f"    └─ {condition}:")
                    hemispheres = list(features[condition].keys())
                    print(f"       Hemispheres: {hemispheres}")

                    # Show feature types for first hemisphere
                    first_hem = hemispheres[0]
                    feature_types = list(features[condition][first_hem].keys())
                    print(f"       Feature types: {feature_types}")

                    # Show scalar feature count
                    if 'stats' in features[condition][first_hem]:
                        n_stats = len(features[condition][first_hem]['stats'])
                        print(f"       └─ stats: {n_stats} scalar features")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error loading {med_state}: {str(e)}")
            result[med_state] = None

    return result


def extract_scalar_features(patient_data: Dict) -> pd.DataFrame:
    """
    Extract all scalar features (stats and persistence entropy) into a DataFrame.

    Args:
        patient_data: Output from load_patient_features()

    Returns:
        DataFrame with columns:
        - patient_id
        - med_state
        - hemisphere ('dominant' or 'nondominant')
        - condition ('hold' or 'resting')
        - All scalar feature values (h0_feature_count, h0_avg_lifespan, etc.)
        - persistence_entropy
    """
    rows = []
    patient_id = patient_data['patient_id']

    for med_state in ['medOn', 'medOff']:
        if patient_data[med_state] is None:
            continue

        features = patient_data[med_state]

        for condition in features.keys():  # 'hold', 'resting'
            for hemisphere in features[condition].keys():  # 'dominant', 'nondominant'
                hem_data = features[condition][hemisphere]

                # Start row with identifiers
                row = {
                    'patient_id': patient_id,
                    'med_state': med_state,
                    'hemisphere': hemisphere,
                    'condition': condition
                }

                # Add all stats (scalar features)
                if 'stats' in hem_data:
                    row.update(hem_data['stats'])

                # Add persistence entropy (one value per homology dimension)
                if 'persistence_entropy' in hem_data:
                    entropy = hem_data['persistence_entropy']
                    # Handle array format - shape is typically (1, 4) for H0, H1, H2, H3
                    if isinstance(entropy, np.ndarray):
                        if entropy.ndim == 2:
                            entropy = entropy.flatten()
                        # Assign each dimension's entropy
                        for i, dim in enumerate(['h0', 'h1', 'h2', 'h3']):
                            if i < len(entropy):
                                row[f'{dim}_persistence_entropy'] = float(entropy[i])
                    else:
                        # Fallback for unexpected format
                        row['persistence_entropy'] = float(entropy)

                rows.append(row)

    return pd.DataFrame(rows)


def extract_array_features(patient_data: Dict, feature_type: str) -> Dict:
    """
    Extract array-based features (landscapes, betti_curves, heat_kernels).

    Args:
        patient_data: Output from load_patient_features()
        feature_type: One of 'persistence_landscape', 'betti_curve', 'heat_kernel'

    Returns:
        Nested dictionary with structure:
        {
            med_state: {
                condition: {
                    hemisphere: array
                }
            }
        }
    """
    if feature_type not in ['persistence_landscape', 'betti_curve', 'heat_kernel']:
        raise ValueError(f"Invalid feature_type: {feature_type}")

    result = {}
    patient_id = patient_data['patient_id']

    for med_state in ['medOn', 'medOff']:
        if patient_data[med_state] is None:
            continue

        result[med_state] = {}
        features = patient_data[med_state]

        for condition in features.keys():
            result[med_state][condition] = {}

            for hemisphere in features[condition].keys():
                hem_data = features[condition][hemisphere]

                if feature_type in hem_data:
                    result[med_state][condition][hemisphere] = hem_data[feature_type]

    return result


def load_all_patients(patients: Optional[List[str]] = None,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Load scalar features from all patients into a single DataFrame.

    Args:
        patients: List of patient IDs to load. If None, loads all patients.
        verbose: If True, print detailed loading information

    Returns:
        DataFrame containing all scalar features from all patients
        Columns: patient_id, med_state, hemisphere, condition, + all feature values
    """
    if patients is None:
        patients = PATIENTS_ALL

    if verbose:
        availability = get_patient_availability()
        n_paired = sum(1 for p in patients if availability[p]['has_medOn'] and availability[p]['has_medOff'])
        n_medoff_only = sum(1 for p in patients if availability[p]['has_medOff'] and not availability[p]['has_medOn'])
        n_medon_only = sum(1 for p in patients if availability[p]['has_medOn'] and not availability[p]['has_medOff'])

        print("\n" + "="*70)
        print("LOADING ALL PATIENTS - OVERVIEW")
        print("="*70)
        print(f"Total patients: {len(patients)}")
        print(f"  • Paired (medOn + medOff): {n_paired}")
        print(f"  • MedOff only: {n_medoff_only}")
        print(f"  • MedOn only: {n_medon_only}")
        print("="*70)

    all_dfs = []

    for i, patient_id in enumerate(patients, 1):
        if verbose:
            print(f"\n[{i}/{len(patients)}] Processing {patient_id}...")

        try:
            patient_data = load_patient_features(patient_id, verbose=False)
            df = extract_scalar_features(patient_data)
            all_dfs.append(df)

            if verbose:
                n_medOn = len(df[df['med_state'] == 'medOn'])
                n_medOff = len(df[df['med_state'] == 'medOff'])
                print(f"  ✓ Loaded: {n_medOn} medOn rows, {n_medOff} medOff rows")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error loading {patient_id}: {str(e)}")

    # Combine all patient data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print("\n" + "="*70)
        print("LOADING COMPLETE - SUMMARY")
        print("="*70)
        print(f"Total rows: {len(combined_df)}")
        print(f"Patients loaded: {combined_df['patient_id'].nunique()}")
        print(f"Medication states: {combined_df['med_state'].unique().tolist()}")
        print(f"Hemispheres: {combined_df['hemisphere'].unique().tolist()}")
        print(f"Conditions: {combined_df['condition'].unique().tolist()}")
        print(f"\nScalar features available: {len(combined_df.columns) - 4}")
        print(f"Feature columns: {[col for col in combined_df.columns if col not in ['patient_id', 'med_state', 'hemisphere', 'condition']][:5]}... (showing first 5)")

        # Show medication state breakdown
        print(f"\nRows by medication state:")
        print(combined_df.groupby('med_state').size())

        print("="*70 + "\n")

    return combined_df


def get_paired_patients_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include patients with both medOn and medOff data.

    Args:
        df: DataFrame from load_all_patients()

    Returns:
        Filtered DataFrame containing only paired patients
    """
    # Find patients that have both medOn and medOff
    patient_states = df.groupby('patient_id')['med_state'].apply(lambda x: set(x))
    paired_patients = patient_states[patient_states.apply(lambda x: {'medOn', 'medOff'}.issubset(x))].index.tolist()

    return df[df['patient_id'].isin(paired_patients)].copy()


def load_persistence_diagrams(patient_id: str, med_state: str,
                              hemisphere: str, condition: str) -> np.ndarray:
    """
    Load raw persistence diagrams for specific patient/condition.

    Args:
        patient_id: Patient identifier
        med_state: 'medOn' or 'medOff'
        hemisphere: 'dominant' or 'nondominant'
        condition: 'hold' or 'resting'

    Returns:
        Persistence diagram array with shape (1, n_features, 3)
        Columns: [birth, death, homology_dimension]
    """
    patient_dir = BASE_DIR / patient_id
    hold_type = PATIENT_HOLD_TYPE[patient_id]

    # Build filename
    if condition == 'hold':
        filename = f"{med_state}_{hemisphere}_{condition}_{hold_type}_diagrams.pkl"
    else:  # resting
        filename = f"{med_state}_{hemisphere}_{condition}_diagrams.pkl"

    diagram_file = patient_dir / filename

    if not diagram_file.exists():
        raise FileNotFoundError(f"Diagram file not found: {diagram_file}")

    with open(diagram_file, 'rb') as f:
        diagram = pickle.load(f)

    return diagram


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("DATA LOADER MODULE - EXAMPLE USAGE")
    print("="*70)

    # Example 1: Load all patients
    print("\n\nExample 1: Loading all patients with verbose output\n")
    df_all = load_all_patients(verbose=True)

    print("\nDataFrame shape:", df_all.shape)
    print("\nFirst few rows:")
    print(df_all.head())

    # Example 2: Get only paired patients
    print("\n\n" + "="*70)
    print("Example 2: Filtering to paired patients only")
    print("="*70)
    df_paired = get_paired_patients_data(df_all)
    print(f"\nPaired patients DataFrame shape: {df_paired.shape}")
    print(f"Patients included: {sorted(df_paired['patient_id'].unique().tolist())}")

    # Example 3: Load single patient
    print("\n\n" + "="*70)
    print("Example 3: Loading single patient (i4oK0F)")
    print("="*70)
    patient_features = load_patient_features('i4oK0F', verbose=True)

    # Example 4: Extract array features
    print("\n\n" + "="*70)
    print("Example 4: Extracting persistence landscapes for patient i4oK0F")
    print("="*70)
    landscapes = extract_array_features(patient_features, 'persistence_landscape')
    print(f"\nMedication states with landscapes: {list(landscapes.keys())}")
    if 'medOff' in landscapes:
        print(f"Conditions in medOff: {list(landscapes['medOff'].keys())}")
        if 'hold' in landscapes['medOff']:
            print(f"Hemispheres in medOff/hold: {list(landscapes['medOff']['hold'].keys())}")
            if 'dominant' in landscapes['medOff']['hold']:
                landscape_shape = landscapes['medOff']['hold']['dominant'].shape
                print(f"Dominant hemisphere landscape shape: {landscape_shape}")
