"""
Statistical Analysis Utilities for Parkinson's Disease TDA Analysis

This module provides statistical testing and analysis functions for comparing
topological features between medOn and medOff medication states.

Key Functions:
- paired_ttest: Paired t-test with Cohen's d effect size
- wilcoxon_test: Non-parametric paired test
- independent_ttest: Independent samples t-test
- compute_effect_size: Cohen's d calculation
- multiple_comparison_correction: FDR and Bonferroni correction
- create_summary_table: Generate results summary
- Array feature summarization for landscapes, Betti curves, heat kernels

Usage:
    from analysis_utils import paired_ttest, multiple_comparison_correction

    # Perform paired t-test
    result = paired_ttest(df, 'h0_feature_count', verbose=True)

    # Apply multiple comparison correction
    corrected = multiple_comparison_correction(p_values, method='fdr_bh')
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union

# Handle scipy version differences for trapz
try:
    from scipy.integrate import trapz
except ImportError:
    # In newer scipy versions, trapz is in numpy
    from numpy import trapz


def compute_effect_size(group1: np.ndarray, group2: np.ndarray,
                       paired: bool = True) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group of observations
        group2: Second group of observations
        paired: If True, uses paired effect size formula (difference / std of differences)
                If False, uses pooled standard deviation

    Returns:
        Cohen's d effect size
        Interpretation: 0.2 (small), 0.5 (medium), 0.8 (large)
    """
    if paired:
        # For paired data: mean difference divided by std of differences
        differences = group1 - group2
        std_diff = np.std(differences, ddof=1)
        if std_diff == 0 or np.isnan(std_diff):
            cohen_d = 0.0  # No variance = no effect
        else:
            cohen_d = np.mean(differences) / std_diff
    else:
        # For independent data: mean difference divided by pooled std
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return cohen_d


def paired_ttest(df: pd.DataFrame, feature_name: str,
                 group_by: Optional[List[str]] = None,
                 verbose: bool = False) -> Dict:
    """
    Perform paired t-test comparing medOn vs medOff for a given feature.

    Only includes patients with both medOn and medOff data.
    Can optionally group by hemisphere and/or condition for separate tests.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        feature_name: Name of the feature to test (e.g., 'h0_feature_count')
        group_by: Optional list of grouping variables ['hemisphere', 'condition']
                  If None, performs single test across all conditions
        verbose: If True, print detailed results

    Returns:
        Dictionary with test results:
        {
            'feature': feature_name,
            'n_patients': int,
            't_statistic': float,
            'p_value': float,
            'cohen_d': float,
            'ci_95': tuple (lower, upper),
            'mean_medOn': float,
            'mean_medOff': float,
            'mean_difference': float,
            'groups': dict (if group_by was used)
        }
    """
    # Filter to patients with both medOn and medOff
    patient_counts = df.groupby('patient_id')['med_state'].apply(lambda x: set(x))
    paired_patients = patient_counts[patient_counts.apply(lambda x: {'medOn', 'medOff'}.issubset(x))].index.tolist()
    df_paired = df[df['patient_id'].isin(paired_patients)].copy()

    if len(paired_patients) == 0:
        return {'error': 'No patients with both medOn and medOff data'}

    # If no grouping, perform single test
    if group_by is None:
        # Average across hemisphere and condition for each patient
        patient_means = df_paired.groupby(['patient_id', 'med_state'])[feature_name].mean().reset_index()

        medOn_values = patient_means[patient_means['med_state'] == 'medOn'][feature_name].values
        medOff_values = patient_means[patient_means['med_state'] == 'medOff'][feature_name].values

        # Ensure same order of patients
        medOn_df = patient_means[patient_means['med_state'] == 'medOn'].set_index('patient_id')
        medOff_df = patient_means[patient_means['med_state'] == 'medOff'].set_index('patient_id')
        common_patients = medOn_df.index.intersection(medOff_df.index)

        medOn_values = medOn_df.loc[common_patients, feature_name].values
        medOff_values = medOff_df.loc[common_patients, feature_name].values

        # Perform paired t-test
        differences = medOn_values - medOff_values

        # Handle zero variance case
        if np.std(differences, ddof=1) == 0:
            t_stat = 0.0
            p_value = 1.0
            ci_95 = (0.0, 0.0)
        else:
            t_stat, p_value = stats.ttest_rel(medOn_values, medOff_values)
            ci_95 = stats.t.interval(0.95, len(differences) - 1,
                                     loc=np.mean(differences),
                                     scale=stats.sem(differences))

        cohen_d = compute_effect_size(medOn_values, medOff_values, paired=True)

        result = {
            'feature': feature_name,
            'n_patients': len(common_patients),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'ci_95': tuple(ci_95),
            'mean_medOn': float(np.mean(medOn_values)),
            'mean_medOff': float(np.mean(medOff_values)),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1))
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"PAIRED T-TEST: {feature_name}")
            print(f"{'='*70}")
            print(f"Number of paired patients: {result['n_patients']}")
            print(f"\nMean (medOn):  {result['mean_medOn']:.4f}")
            print(f"Mean (medOff): {result['mean_medOff']:.4f}")
            print(f"Mean difference (medOn - medOff): {result['mean_difference']:.4f}")
            print(f"\nt-statistic: {result['t_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.4f}")
            print(f"Cohen's d: {result['cohen_d']:.4f} ({interpret_effect_size(result['cohen_d'])})")
            print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
            print(f"{'='*70}\n")

        return result

    # If grouping, perform separate tests for each group
    else:
        results = {'feature': feature_name, 'groups': {}}

        for group_values, group_df in df_paired.groupby(group_by):
            group_name = group_values if isinstance(group_values, str) else '_'.join(group_values)

            # Get medOn and medOff values for this group
            medOn_df = group_df[group_df['med_state'] == 'medOn'].set_index('patient_id')
            medOff_df = group_df[group_df['med_state'] == 'medOff'].set_index('patient_id')
            common_patients = medOn_df.index.intersection(medOff_df.index)

            if len(common_patients) < 3:  # Need at least 3 pairs for meaningful test
                continue

            medOn_values = medOn_df.loc[common_patients, feature_name].values
            medOff_values = medOff_df.loc[common_patients, feature_name].values

            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(medOn_values, medOff_values)
            cohen_d = compute_effect_size(medOn_values, medOff_values, paired=True)

            differences = medOn_values - medOff_values
            ci_95 = stats.t.interval(0.95, len(differences) - 1,
                                     loc=np.mean(differences),
                                     scale=stats.sem(differences))

            results['groups'][group_name] = {
                'n_patients': len(common_patients),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohen_d': float(cohen_d),
                'ci_95': tuple(ci_95),
                'mean_medOn': float(np.mean(medOn_values)),
                'mean_medOff': float(np.mean(medOff_values)),
                'mean_difference': float(np.mean(differences))
            }

            if verbose:
                print(f"\nGroup: {group_name}")
                print(f"  n={len(common_patients)}, t={t_stat:.3f}, p={p_value:.4f}, d={cohen_d:.3f}")

        return results


def wilcoxon_test(df: pd.DataFrame, feature_name: str,
                  group_by: Optional[List[str]] = None,
                  verbose: bool = False) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired test).

    Use when data is not normally distributed or has outliers.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        feature_name: Name of the feature to test
        group_by: Optional grouping variables
        verbose: If True, print detailed results

    Returns:
        Dictionary with test results (similar to paired_ttest but with W statistic)
    """
    # Similar structure to paired_ttest but using Wilcoxon test
    patient_counts = df.groupby('patient_id')['med_state'].apply(lambda x: set(x))
    paired_patients = patient_counts[patient_counts.apply(lambda x: {'medOn', 'medOff'}.issubset(x))].index.tolist()
    df_paired = df[df['patient_id'].isin(paired_patients)].copy()

    if len(paired_patients) == 0:
        return {'error': 'No patients with both medOn and medOff data'}

    if group_by is None:
        patient_means = df_paired.groupby(['patient_id', 'med_state'])[feature_name].mean().reset_index()

        medOn_df = patient_means[patient_means['med_state'] == 'medOn'].set_index('patient_id')
        medOff_df = patient_means[patient_means['med_state'] == 'medOff'].set_index('patient_id')
        common_patients = medOn_df.index.intersection(medOff_df.index)

        medOn_values = medOn_df.loc[common_patients, feature_name].values
        medOff_values = medOff_df.loc[common_patients, feature_name].values

        # Perform Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(medOn_values, medOff_values)

        # Effect size for Wilcoxon: r = Z / sqrt(N)
        z_score = stats.norm.ppf(1 - p_value / 2) * np.sign(np.median(medOn_values - medOff_values))
        effect_size_r = z_score / np.sqrt(len(common_patients))

        result = {
            'feature': feature_name,
            'n_patients': len(common_patients),
            'w_statistic': float(w_stat),
            'p_value': float(p_value),
            'effect_size_r': float(effect_size_r),
            'median_medOn': float(np.median(medOn_values)),
            'median_medOff': float(np.median(medOff_values)),
            'median_difference': float(np.median(medOn_values - medOff_values))
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"WILCOXON SIGNED-RANK TEST: {feature_name}")
            print(f"{'='*70}")
            print(f"Number of paired patients: {result['n_patients']}")
            print(f"\nMedian (medOn):  {result['median_medOn']:.4f}")
            print(f"Median (medOff): {result['median_medOff']:.4f}")
            print(f"Median difference: {result['median_difference']:.4f}")
            print(f"\nW-statistic: {result['w_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.4f}")
            print(f"Effect size (r): {result['effect_size_r']:.4f}")
            print(f"{'='*70}\n")

        return result

    # Grouped version
    else:
        results = {'feature': feature_name, 'groups': {}}

        for group_values, group_df in df_paired.groupby(group_by):
            group_name = group_values if isinstance(group_values, str) else '_'.join(group_values)

            medOn_df = group_df[group_df['med_state'] == 'medOn'].set_index('patient_id')
            medOff_df = group_df[group_df['med_state'] == 'medOff'].set_index('patient_id')
            common_patients = medOn_df.index.intersection(medOff_df.index)

            if len(common_patients) < 3:
                continue

            medOn_values = medOn_df.loc[common_patients, feature_name].values
            medOff_values = medOff_df.loc[common_patients, feature_name].values

            w_stat, p_value = stats.wilcoxon(medOn_values, medOff_values)
            z_score = stats.norm.ppf(1 - p_value / 2) * np.sign(np.median(medOn_values - medOff_values))
            effect_size_r = z_score / np.sqrt(len(common_patients))

            results['groups'][group_name] = {
                'n_patients': len(common_patients),
                'w_statistic': float(w_stat),
                'p_value': float(p_value),
                'effect_size_r': float(effect_size_r),
                'median_medOn': float(np.median(medOn_values)),
                'median_medOff': float(np.median(medOff_values)),
                'median_difference': float(np.median(medOn_values - medOff_values))
            }

        return results


def independent_ttest(df: pd.DataFrame, feature_name: str,
                     verbose: bool = False) -> Dict:
    """
    Perform independent samples t-test comparing medOn vs medOff.

    Uses all available patients (not just paired).
    Less powerful than paired test but includes more data.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        feature_name: Name of the feature to test
        verbose: If True, print detailed results

    Returns:
        Dictionary with test results
    """
    medOn_values = df[df['med_state'] == 'medOn'][feature_name].values
    medOff_values = df[df['med_state'] == 'medOff'][feature_name].values

    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(medOn_values, medOff_values)
    cohen_d = compute_effect_size(medOn_values, medOff_values, paired=False)

    result = {
        'feature': feature_name,
        'n_medOn': len(medOn_values),
        'n_medOff': len(medOff_values),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohen_d': float(cohen_d),
        'mean_medOn': float(np.mean(medOn_values)),
        'mean_medOff': float(np.mean(medOff_values)),
        'mean_difference': float(np.mean(medOn_values) - np.mean(medOff_values))
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEPENDENT T-TEST: {feature_name}")
        print(f"{'='*70}")
        print(f"n (medOn): {result['n_medOn']}, n (medOff): {result['n_medOff']}")
        print(f"\nMean (medOn):  {result['mean_medOn']:.4f}")
        print(f"Mean (medOff): {result['mean_medOff']:.4f}")
        print(f"Mean difference: {result['mean_difference']:.4f}")
        print(f"\nt-statistic: {result['t_statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
        print(f"Cohen's d: {result['cohen_d']:.4f} ({interpret_effect_size(result['cohen_d'])})")
        print(f"{'='*70}\n")

    return result


def multiple_comparison_correction(p_values: Union[List[float], np.ndarray],
                                   method: str = 'fdr_bh',
                                   alpha: float = 0.05) -> Dict:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List or array of p-values
        method: Correction method
                'bonferroni': Bonferroni correction (most conservative)
                'fdr_bh': Benjamini-Hochberg FDR (less conservative)
                'fdr_by': Benjamini-Yekutieli FDR
                'holm': Holm-Bonferroni (step-down)
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with:
        - 'reject': Boolean array indicating which tests are significant
        - 'p_corrected': Corrected p-values
        - 'alpha_corrected': Corrected alpha level (for Bonferroni)
        - 'method': Method used
    """
    p_values = np.array(p_values)

    # Apply correction
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=method
    )

    result = {
        'method': method,
        'alpha': alpha,
        'n_tests': len(p_values),
        'reject': reject,
        'p_corrected': p_corrected,
        'n_significant': int(np.sum(reject)),
        'alpha_bonferroni': alpha_bonf if method == 'bonferroni' else None
    }

    return result


def interpret_effect_size(cohen_d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        cohen_d: Cohen's d value

    Returns:
        String interpretation: 'negligible', 'small', 'medium', 'large', 'very large'
    """
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    elif abs_d < 1.3:
        return 'large'
    else:
        return 'very large'


def create_summary_table(results_list: List[Dict],
                        sort_by: str = 'p_value',
                        include_corrected: bool = False) -> pd.DataFrame:
    """
    Create a summary table from multiple test results.

    Args:
        results_list: List of result dictionaries from paired_ttest or similar
        sort_by: Column to sort by ('p_value', 'cohen_d', 'feature')
        include_corrected: If True, apply FDR correction and include column

    Returns:
        DataFrame with summary of all tests
    """
    summary_data = []

    for result in results_list:
        if 'error' in result:
            continue

        row = {
            'feature': result['feature'],
            'n_patients': result.get('n_patients', result.get('n_medOn', 0)),
            'mean_medOn': result.get('mean_medOn', np.nan),
            'mean_medOff': result.get('mean_medOff', np.nan),
            'mean_difference': result.get('mean_difference', np.nan),
            't_statistic': result.get('t_statistic', result.get('w_statistic', np.nan)),
            'p_value': result['p_value'],
            'cohen_d': result.get('cohen_d', np.nan),
            'effect_size_interpretation': interpret_effect_size(result.get('cohen_d', 0))
        }

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Apply multiple comparison correction if requested
    if include_corrected and len(df) > 0:
        correction = multiple_comparison_correction(df['p_value'].values, method='fdr_bh')
        df['p_corrected'] = correction['p_corrected']
        df['significant_corrected'] = correction['reject']

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by)

    return df


# ============================================================================
# ARRAY FEATURE SUMMARIZATION
# ============================================================================

def summarize_persistence_landscape(landscape: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a persistence landscape.

    Args:
        landscape: Persistence landscape array, typically shape (1, n_dimensions, n_samples)
                   or (n_dimensions, n_samples)

    Returns:
        Dictionary with summary metrics:
        - 'l1_norm': L1 norm (sum of absolute values)
        - 'l2_norm': L2 norm (Euclidean norm)
        - 'linf_norm': L-infinity norm (maximum absolute value)
        - 'auc': Area under the curve (integral)
        - 'peak_value': Maximum value
        - 'peak_location': Location of peak (index)
        - Per-dimension summaries if multi-dimensional
    """
    # Handle shape variations
    if landscape.ndim == 3:
        landscape = landscape[0]  # Remove batch dimension

    summary = {}

    if landscape.ndim == 2:
        # Multiple homology dimensions
        n_dims = landscape.shape[0]

        for dim in range(n_dims):
            dim_landscape = landscape[dim, :]

            summary[f'h{dim}_l1_norm'] = float(np.sum(np.abs(dim_landscape)))
            summary[f'h{dim}_l2_norm'] = float(np.linalg.norm(dim_landscape))
            summary[f'h{dim}_linf_norm'] = float(np.max(np.abs(dim_landscape)))
            summary[f'h{dim}_auc'] = float(trapz(dim_landscape))
            summary[f'h{dim}_peak_value'] = float(np.max(dim_landscape))
            summary[f'h{dim}_peak_location'] = int(np.argmax(dim_landscape))
            summary[f'h{dim}_mean'] = float(np.mean(dim_landscape))
            summary[f'h{dim}_std'] = float(np.std(dim_landscape))
    else:
        # Single dimension
        summary['l1_norm'] = float(np.sum(np.abs(landscape)))
        summary['l2_norm'] = float(np.linalg.norm(landscape))
        summary['linf_norm'] = float(np.max(np.abs(landscape)))
        summary['auc'] = float(trapz(landscape))
        summary['peak_value'] = float(np.max(landscape))
        summary['peak_location'] = int(np.argmax(landscape))
        summary['mean'] = float(np.mean(landscape))
        summary['std'] = float(np.std(landscape))

    return summary


def summarize_betti_curve(betti_curve: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a Betti curve.

    Args:
        betti_curve: Betti curve array, typically shape (1, n_dimensions, n_samples)

    Returns:
        Dictionary with summary metrics:
        - 'auc': Area under the curve
        - 'max_betti': Maximum Betti number
        - 'max_betti_location': Location of maximum
        - 'centroid': Centroid of the curve (first moment)
        - Per-dimension summaries if multi-dimensional
    """
    # Handle shape variations
    if betti_curve.ndim == 3:
        betti_curve = betti_curve[0]

    summary = {}

    if betti_curve.ndim == 2:
        # Multiple homology dimensions
        n_dims = betti_curve.shape[0]

        for dim in range(n_dims):
            curve = betti_curve[dim, :]

            summary[f'h{dim}_auc'] = float(trapz(curve))
            summary[f'h{dim}_max_betti'] = float(np.max(curve))
            summary[f'h{dim}_max_betti_location'] = int(np.argmax(curve))

            # Centroid: weighted average of indices
            if np.sum(curve) > 0:
                indices = np.arange(len(curve))
                summary[f'h{dim}_centroid'] = float(np.sum(indices * curve) / np.sum(curve))
            else:
                summary[f'h{dim}_centroid'] = 0.0

            summary[f'h{dim}_mean'] = float(np.mean(curve))
            summary[f'h{dim}_std'] = float(np.std(curve))
    else:
        # Single dimension
        summary['auc'] = float(trapz(betti_curve))
        summary['max_betti'] = float(np.max(betti_curve))
        summary['max_betti_location'] = int(np.argmax(betti_curve))

        if np.sum(betti_curve) > 0:
            indices = np.arange(len(betti_curve))
            summary['centroid'] = float(np.sum(indices * betti_curve) / np.sum(betti_curve))
        else:
            summary['centroid'] = 0.0

        summary['mean'] = float(np.mean(betti_curve))
        summary['std'] = float(np.std(betti_curve))

    return summary


def summarize_heat_kernel(heat_kernel: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a heat kernel signature.

    Args:
        heat_kernel: Heat kernel array, typically shape (1, n_dimensions, n, n)

    Returns:
        Dictionary with summary metrics:
        - 'frobenius_norm': Frobenius norm of the matrix
        - 'trace': Trace (sum of diagonal elements)
        - 'mean': Mean value
        - 'std': Standard deviation
        - Per-dimension summaries if multi-dimensional
    """
    # Handle shape variations
    if heat_kernel.ndim == 4:
        heat_kernel = heat_kernel[0]  # Remove batch dimension

    summary = {}

    if heat_kernel.ndim == 3:
        # Multiple homology dimensions
        n_dims = heat_kernel.shape[0]

        for dim in range(n_dims):
            kernel = heat_kernel[dim, :, :]

            summary[f'h{dim}_frobenius_norm'] = float(np.linalg.norm(kernel, 'fro'))
            summary[f'h{dim}_trace'] = float(np.trace(kernel))
            summary[f'h{dim}_mean'] = float(np.mean(kernel))
            summary[f'h{dim}_std'] = float(np.std(kernel))
            summary[f'h{dim}_max'] = float(np.max(kernel))
            summary[f'h{dim}_min'] = float(np.min(kernel))
    else:
        # Single dimension matrix
        summary['frobenius_norm'] = float(np.linalg.norm(heat_kernel, 'fro'))
        summary['trace'] = float(np.trace(heat_kernel))
        summary['mean'] = float(np.mean(heat_kernel))
        summary['std'] = float(np.std(heat_kernel))
        summary['max'] = float(np.max(heat_kernel))
        summary['min'] = float(np.min(heat_kernel))

    return summary


# Example usage and testing
if __name__ == "__main__":
    from data_loader import load_all_patients

    print("="*70)
    print("ANALYSIS UTILS MODULE - EXAMPLE USAGE")
    print("="*70)

    # Load data
    print("\nLoading patient data...")
    df = load_all_patients(verbose=False)
    print(f"Loaded {len(df)} rows from {df['patient_id'].nunique()} patients")

    # Example 1: Paired t-test for a single feature
    print("\n\nExample 1: Paired t-test for H0 feature count\n")
    result1 = paired_ttest(df, 'h0_feature_count', verbose=True)

    # Example 2: Paired t-test grouped by hemisphere
    print("\n\nExample 2: Paired t-test for H1 persistence entropy (by hemisphere)\n")
    result2 = paired_ttest(df, 'h1_persistence_entropy', group_by=['hemisphere'], verbose=True)

    # Example 3: Run tests on multiple features
    print("\n\nExample 3: Testing multiple features\n")
    print("="*70)
    features_to_test = ['h0_feature_count', 'h1_feature_count', 'h0_persistence_entropy', 'h1_persistence_entropy']

    results = []
    for feature in features_to_test:
        result = paired_ttest(df, feature, verbose=False)
        results.append(result)

    # Create summary table
    summary = create_summary_table(results, include_corrected=True)
    print("\nSummary Table:")
    print(summary.to_string())

    # Example 4: Wilcoxon test
    print("\n\nExample 4: Wilcoxon test for H0 average lifespan\n")
    result4 = wilcoxon_test(df, 'h0_avg_lifespan', verbose=True)

    print("\n✓ All examples completed successfully!")
