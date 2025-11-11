#!/usr/bin/env python3
"""
Example: Load and Use Aggregated Features

This script demonstrates how to load aggregated features and prepare them for analysis.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_aggregated_features(patient_folder):
    """Load aggregated features for a single patient."""
    agg_file = Path(patient_folder) / 'aggregated_features.pkl'

    if not agg_file.exists():
        raise FileNotFoundError(f"Aggregated features not found: {agg_file}")

    with open(agg_file, 'rb') as f:
        data = pickle.load(f)

    return data


def extract_scalar_features(aggregated_data, patient_id):
    """
    Extract all scalar features into a flat dataframe.

    Args:
        aggregated_data: Dictionary from aggregated_features.pkl
        patient_id: Patient identifier

    Returns:
        DataFrame with one row per patient-medstate-hemisphere-condition combination
    """
    rows = []

    for med_state in ['medOn', 'medOff']:
        if med_state not in aggregated_data:
            continue

        for condition_key in ['left_hold', 'left_resting', 'right_hold', 'right_resting']:
            if condition_key not in aggregated_data[med_state]:
                continue

            features = aggregated_data[med_state][condition_key]

            # Extract only scalar features (not arrays or diagrams)
            scalar_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    scalar_features[key] = value
                elif key == 'hemisphere':
                    scalar_features['hemisphere'] = value
                elif key == 'condition':
                    scalar_features['condition'] = value

            # Add metadata
            row = {
                'patient': patient_id,
                'med_state': med_state,
                **scalar_features
            }

            rows.append(row)

    return pd.DataFrame(rows)


def create_analysis_dataframe(patient_folders):
    """
    Create a combined dataframe from all patients for analysis.

    Args:
        patient_folders: List of patient folder paths

    Returns:
        DataFrame with all patients' aggregated features
    """
    all_dfs = []

    for patient_folder in patient_folders:
        patient_folder = Path(patient_folder)
        patient_id = patient_folder.name

        try:
            data = load_aggregated_features(patient_folder)
            df = extract_scalar_features(data, patient_id)
            all_dfs.append(df)
            print(f"✓ Loaded {patient_id}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Failed to load {patient_id}: {e}")

    if not all_dfs:
        raise ValueError("No data loaded successfully")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def main():
    """Example usage."""
    print("="*80)
    print("EXAMPLE: Loading and Using Aggregated Features")
    print("="*80)

    # Example 1: Load single patient
    print("\n1. Loading single patient (i4oK0F)...")
    data = load_aggregated_features('./i4oK0F/')

    print(f"\nAvailable medication states: {list(data.keys())}")
    print(f"Available conditions (medOff): {list(data['medOff'].keys())}")

    # Access specific features
    medOn_left_hold = data['medOn']['left_hold']
    medOff_left_hold = data['medOff']['left_hold']

    print(f"\nSample features from medOn left_hold:")
    for key in list(medOn_left_hold.keys())[:5]:
        value = medOn_left_hold[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Compare medOn vs medOff
    print(f"\nComparison (left hold):")
    print(f"  MedOn persistence entropy: {medOn_left_hold['persistence_entropy_mean']:.4f} ± {medOn_left_hold['persistence_entropy_std']:.4f}")
    print(f"  MedOff persistence entropy: {medOff_left_hold['persistence_entropy_mean']:.4f} ± {medOff_left_hold['persistence_entropy_std']:.4f}")
    print(f"  Difference: {medOn_left_hold['persistence_entropy_mean'] - medOff_left_hold['persistence_entropy_mean']:.4f}")

    # Example 2: Create dataframe for analysis
    print(f"\n2. Creating analysis dataframe...")

    # Find all patient folders with aggregated features
    patient_folders = [p.parent for p in Path('.').glob('*/aggregated_features.pkl')]

    if patient_folders:
        print(f"\nFound {len(patient_folders)} patient(s) with aggregated features")

        df = create_analysis_dataframe(patient_folders)

        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nFirst few columns:")
        print(df.columns[:15].tolist())

        # Example analysis: Compare medOn vs medOff
        print(f"\n3. Example statistical comparison...")

        # Filter for a specific condition (e.g., left hold)
        left_hold_df = df[(df['hemisphere'] == 'left') & (df['condition'] == 'hold')]

        # Separate by medication state
        medOn_data = left_hold_df[left_hold_df['med_state'] == 'medOn']
        medOff_data = left_hold_df[left_hold_df['med_state'] == 'medOff']

        print(f"\nLeft hold condition:")
        print(f"  MedOn: {len(medOn_data)} patients")
        print(f"  MedOff: {len(medOff_data)} patients")

        # Compare a specific feature
        if 'persistence_entropy_mean' in df.columns:
            medOn_entropy = medOn_data['persistence_entropy_mean'].values
            medOff_entropy = medOff_data['persistence_entropy_mean'].values

            print(f"\nPersistence Entropy (left hold):")
            print(f"  MedOn:  {np.mean(medOn_entropy):.4f} ± {np.std(medOn_entropy):.4f}")
            print(f"  MedOff: {np.mean(medOff_entropy):.4f} ± {np.std(medOff_entropy):.4f}")

            # Paired t-test (if same number of patients)
            if len(medOn_entropy) == len(medOff_entropy):
                from scipy import stats
                t_stat, p_value = stats.ttest_rel(medOn_entropy, medOff_entropy)
                print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

                if p_value < 0.05:
                    print(f"  → Significant difference!")
                else:
                    print(f"  → No significant difference")

        # Save dataframe for further analysis
        output_file = 'combined_aggregated_features.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved combined dataframe to: {output_file}")

    else:
        print("\nNo patient folders with aggregated features found.")
        print("Run 'python aggregate_features.py <patient_folder>' first.")

    print(f"\n{'='*80}")
    print("Example complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
