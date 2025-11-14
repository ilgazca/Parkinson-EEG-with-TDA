"""
Visualization Utilities for Parkinson's Disease TDA Analysis

This module provides plotting functions for comparing topological features
between medOn and medOff medication states.

Key Functions:
- plot_paired_scatter: Scatter plot comparing medOn vs medOff per patient
- plot_distribution_comparison: Box/violin plots for feature distributions
- plot_forest: Forest plot showing effect sizes with confidence intervals
- plot_raincloud: Combined distribution visualization
- plot_landscape_comparison: Mean landscapes with error bands
- plot_betti_comparison: Mean Betti curves with error bands
- plot_heatmap_comparison: Side-by-side heatmaps for heat kernels
- plot_summary_panel: Multi-panel figure for comprehensive results

Usage:
    from visualization_utils import plot_paired_scatter, plot_distribution_comparison

    # Create paired scatter plot
    fig = plot_paired_scatter(df, 'h0_feature_count', save_path='results/figures/')

    # Create distribution comparison
    fig = plot_distribution_comparison(df, 'h1_persistence_entropy', plot_type='violin')
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import warnings

# Set default style
plt.style.use('default')
sns.set_palette("husl")

# Default figure parameters
DEFAULT_FIGSIZE = (8, 6)
DEFAULT_DPI = 100
DEFAULT_COLORS = {
    'medOn': '#3498db',   # Blue
    'medOff': '#e74c3c',  # Red
    'paired_line': '#95a5a6'  # Gray for connecting lines
}


def plot_paired_scatter(df: pd.DataFrame,
                       feature_name: str,
                       group_by: Optional[str] = None,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
                       show_line: bool = True,
                       show_identity: bool = True) -> plt.Figure:
    """
    Create paired scatter plot comparing medOn vs medOff for each patient.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        feature_name: Name of feature to plot
        group_by: Optional grouping ('hemisphere' or 'condition')
        title: Plot title (auto-generated if None)
        save_path: Path to save figure (if provided)
        figsize: Figure size (width, height)
        show_line: If True, connect paired points with lines
        show_identity: If True, show diagonal identity line

    Returns:
        Matplotlib figure object
    """
    # Filter to paired patients
    patient_counts = df.groupby('patient_id')['med_state'].apply(lambda x: set(x))
    paired_patients = patient_counts[patient_counts.apply(lambda x: {'medOn', 'medOff'}.issubset(x))].index.tolist()
    df_paired = df[df['patient_id'].isin(paired_patients)].copy()

    if len(paired_patients) == 0:
        raise ValueError("No patients with both medOn and medOff data")

    # If grouping, create subplots
    if group_by is not None:
        groups = sorted(df_paired[group_by].unique())
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(figsize[0] * n_groups, figsize[1]))
        if n_groups == 1:
            axes = [axes]

        for i, group in enumerate(groups):
            ax = axes[i]
            group_df = df_paired[df_paired[group_by] == group]

            _plot_paired_scatter_single(group_df, feature_name, ax,
                                       title=f"{group.capitalize()}" if title is None else title,
                                       show_line=show_line, show_identity=show_identity)
    else:
        # Single plot
        fig, ax = plt.subplots(figsize=figsize)
        _plot_paired_scatter_single(df_paired, feature_name, ax,
                                   title=title or f"MedOn vs MedOff: {feature_name}",
                                   show_line=show_line, show_identity=show_identity)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"paired_scatter_{feature_name}.png"
        if group_by:
            filename = f"paired_scatter_{feature_name}_by_{group_by}.png"
        fig.savefig(save_path / filename, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def _plot_paired_scatter_single(df: pd.DataFrame, feature_name: str, ax: plt.Axes,
                                title: str, show_line: bool, show_identity: bool):
    """Helper function for plotting single paired scatter plot."""
    # Average across conditions/hemispheres for each patient
    patient_means = df.groupby(['patient_id', 'med_state'])[feature_name].mean().reset_index()

    medOn_df = patient_means[patient_means['med_state'] == 'medOn'].set_index('patient_id')
    medOff_df = patient_means[patient_means['med_state'] == 'medOff'].set_index('patient_id')
    common_patients = medOn_df.index.intersection(medOff_df.index)

    medOn_values = medOn_df.loc[common_patients, feature_name].values
    medOff_values = medOff_df.loc[common_patients, feature_name].values

    # Plot connecting lines
    if show_line:
        for i in range(len(common_patients)):
            ax.plot([medOn_values[i], medOff_values[i]],
                   [medOn_values[i], medOff_values[i]],
                   color=DEFAULT_COLORS['paired_line'], alpha=0.3, linewidth=1, zorder=1)

    # Plot identity line
    if show_identity:
        lim_min = min(medOn_values.min(), medOff_values.min())
        lim_max = max(medOn_values.max(), medOff_values.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
               'k--', alpha=0.3, linewidth=1, zorder=0, label='Identity')

    # Plot points
    ax.scatter(medOn_values, medOff_values, s=100, alpha=0.7,
              edgecolors='black', linewidth=1, zorder=2)

    # Labels and title
    ax.set_xlabel('MedOn', fontsize=12, fontweight='bold')
    ax.set_ylabel('MedOff', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add n=X annotation
    ax.text(0.05, 0.95, f'n={len(common_patients)}',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_distribution_comparison(df: pd.DataFrame,
                                 feature_name: str,
                                 plot_type: str = 'violin',
                                 group_by: Optional[str] = None,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> plt.Figure:
    """
    Create box plot or violin plot comparing medOn vs medOff distributions.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        feature_name: Name of feature to plot
        plot_type: 'box', 'violin', or 'both'
        group_by: Optional grouping ('hemisphere' or 'condition')
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if group_by is not None:
        # Create grouped plot
        if plot_type == 'violin':
            sns.violinplot(data=df, x=group_by, y=feature_name, hue='med_state',
                          palette={'medOn': DEFAULT_COLORS['medOn'],
                                  'medOff': DEFAULT_COLORS['medOff']},
                          ax=ax, split=False)
        elif plot_type == 'box':
            sns.boxplot(data=df, x=group_by, y=feature_name, hue='med_state',
                       palette={'medOn': DEFAULT_COLORS['medOn'],
                               'medOff': DEFAULT_COLORS['medOff']},
                       ax=ax)
        elif plot_type == 'both':
            sns.violinplot(data=df, x=group_by, y=feature_name, hue='med_state',
                          palette={'medOn': DEFAULT_COLORS['medOn'],
                                  'medOff': DEFAULT_COLORS['medOff']},
                          ax=ax, split=False, inner=None, alpha=0.4)
            sns.boxplot(data=df, x=group_by, y=feature_name, hue='med_state',
                       palette={'medOn': DEFAULT_COLORS['medOn'],
                               'medOff': DEFAULT_COLORS['medOff']},
                       ax=ax, width=0.3)
    else:
        # Simple comparison without grouping
        if plot_type == 'violin':
            sns.violinplot(data=df, x='med_state', y=feature_name,
                          palette={'medOn': DEFAULT_COLORS['medOn'],
                                  'medOff': DEFAULT_COLORS['medOff']},
                          ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=df, x='med_state', y=feature_name,
                       palette={'medOn': DEFAULT_COLORS['medOn'],
                               'medOff': DEFAULT_COLORS['medOff']},
                       ax=ax)
        elif plot_type == 'both':
            sns.violinplot(data=df, x='med_state', y=feature_name,
                          palette={'medOn': DEFAULT_COLORS['medOn'],
                                  'medOff': DEFAULT_COLORS['medOff']},
                          ax=ax, inner=None, alpha=0.4)
            sns.boxplot(data=df, x='med_state', y=feature_name,
                       palette={'medOn': DEFAULT_COLORS['medOn'],
                               'medOff': DEFAULT_COLORS['medOff']},
                       ax=ax, width=0.3)

    # Labels and title
    ax.set_ylabel(feature_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Medication State' if group_by is None else group_by.capitalize(),
                  fontsize=12, fontweight='bold')
    ax.set_title(title or f"Distribution Comparison: {feature_name}",
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"distribution_{plot_type}_{feature_name}.png"
        if group_by:
            filename = f"distribution_{plot_type}_{feature_name}_by_{group_by}.png"
        fig.savefig(save_path / filename, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def plot_forest(results_df: pd.DataFrame,
               effect_col: str = 'cohen_d',
               ci_cols: Tuple[str, str] = ('ci_95',),
               p_col: str = 'p_value',
               alpha: float = 0.05,
               title: Optional[str] = None,
               save_path: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create forest plot showing effect sizes with confidence intervals.

    Args:
        results_df: DataFrame with test results (from analysis_utils.create_summary_table)
        effect_col: Column name for effect size
        ci_cols: Column name(s) for confidence intervals
        p_col: Column name for p-values
        alpha: Significance level for coloring
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    df = results_df.copy()

    # Extract CI if it's stored as tuple in single column
    if len(ci_cols) == 1 and ci_cols[0] in df.columns:
        df['ci_lower'] = df[ci_cols[0]].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        df['ci_upper'] = df[ci_cols[0]].apply(lambda x: x[1] if isinstance(x, tuple) else x)
    elif len(ci_cols) == 2:
        df['ci_lower'] = df[ci_cols[0]]
        df['ci_upper'] = df[ci_cols[1]]
    else:
        # No CI available, use effect size only
        df['ci_lower'] = df[effect_col]
        df['ci_upper'] = df[effect_col]

    # Sort by effect size
    df = df.sort_values(effect_col, ascending=True).reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color by significance
    colors = [DEFAULT_COLORS['medOff'] if p < alpha else 'gray'
             for p in df[p_col]]

    # Plot CIs
    for i, row in df.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
               color=colors[i], linewidth=2, alpha=0.7)

    # Plot effect sizes
    ax.scatter(df[effect_col], range(len(df)),
              s=100, color=colors, edgecolors='black', linewidth=1, zorder=3)

    # Reference line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Effect size interpretation lines
    for val, label in [(-0.8, 'Large'), (-0.5, 'Medium'), (-0.2, 'Small'),
                       (0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax.axvline(x=val, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'], fontsize=10)
    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
    ax.set_title(title or "Forest Plot: Effect Sizes with 95% CI",
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    sig_patch = mpatches.Patch(color=DEFAULT_COLORS['medOff'], label=f'Significant (p < {alpha})')
    ns_patch = mpatches.Patch(color='gray', label='Not significant')
    ax.legend(handles=[sig_patch, ns_patch], loc='best')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "forest_plot.png", dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def plot_landscape_comparison(landscapes_medOn: np.ndarray,
                              landscapes_medOff: np.ndarray,
                              homology_dim: int = 1,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> plt.Figure:
    """
    Plot mean persistence landscapes with error bands for medOn vs medOff.

    Args:
        landscapes_medOn: Array of landscapes for medOn (n_patients, n_dims, n_samples)
        landscapes_medOff: Array of landscapes for medOff
        homology_dim: Which homology dimension to plot (0, 1, 2, or 3)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle different input shapes
    if landscapes_medOn.ndim == 4:
        landscapes_medOn = landscapes_medOn[:, 0, :, :]  # Remove batch dimension
    if landscapes_medOff.ndim == 4:
        landscapes_medOff = landscapes_medOff[:, 0, :, :]

    # Extract specific homology dimension
    if landscapes_medOn.ndim == 3:
        landscapes_medOn = landscapes_medOn[:, homology_dim, :]
        landscapes_medOff = landscapes_medOff[:, homology_dim, :]

    # Compute mean and SEM
    mean_medOn = np.mean(landscapes_medOn, axis=0)
    sem_medOn = np.std(landscapes_medOn, axis=0) / np.sqrt(len(landscapes_medOn))

    mean_medOff = np.mean(landscapes_medOff, axis=0)
    sem_medOff = np.std(landscapes_medOff, axis=0) / np.sqrt(len(landscapes_medOff))

    x = np.arange(len(mean_medOn))

    # Plot mean curves with error bands
    ax.plot(x, mean_medOn, color=DEFAULT_COLORS['medOn'], linewidth=2, label='MedOn')
    ax.fill_between(x, mean_medOn - sem_medOn, mean_medOn + sem_medOn,
                    color=DEFAULT_COLORS['medOn'], alpha=0.3)

    ax.plot(x, mean_medOff, color=DEFAULT_COLORS['medOff'], linewidth=2, label='MedOff')
    ax.fill_between(x, mean_medOff - sem_medOff, mean_medOff + sem_medOff,
                    color=DEFAULT_COLORS['medOff'], alpha=0.3)

    # Labels
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Landscape Value', fontsize=12, fontweight='bold')
    ax.set_title(title or f"Persistence Landscape Comparison (H{homology_dim})",
                fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"landscape_comparison_h{homology_dim}.png",
                   dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def plot_betti_comparison(betti_medOn: np.ndarray,
                         betti_medOff: np.ndarray,
                         homology_dim: int = 1,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> plt.Figure:
    """
    Plot mean Betti curves with error bands for medOn vs medOff.

    Args:
        betti_medOn: Array of Betti curves for medOn (n_patients, n_dims, n_samples)
        betti_medOff: Array of Betti curves for medOff
        homology_dim: Which homology dimension to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle different input shapes
    if betti_medOn.ndim == 4:
        betti_medOn = betti_medOn[:, 0, :, :]
    if betti_medOff.ndim == 4:
        betti_medOff = betti_medOff[:, 0, :, :]

    # Extract specific homology dimension
    if betti_medOn.ndim == 3:
        betti_medOn = betti_medOn[:, homology_dim, :]
        betti_medOff = betti_medOff[:, homology_dim, :]

    # Compute mean and SEM
    mean_medOn = np.mean(betti_medOn, axis=0)
    sem_medOn = np.std(betti_medOn, axis=0) / np.sqrt(len(betti_medOn))

    mean_medOff = np.mean(betti_medOff, axis=0)
    sem_medOff = np.std(betti_medOff, axis=0) / np.sqrt(len(betti_medOff))

    x = np.arange(len(mean_medOn))

    # Plot curves with error bands
    ax.plot(x, mean_medOn, color=DEFAULT_COLORS['medOn'], linewidth=2, label='MedOn')
    ax.fill_between(x, mean_medOn - sem_medOn, mean_medOn + sem_medOn,
                    color=DEFAULT_COLORS['medOn'], alpha=0.3)

    ax.plot(x, mean_medOff, color=DEFAULT_COLORS['medOff'], linewidth=2, label='MedOff')
    ax.fill_between(x, mean_medOff - sem_medOff, mean_medOff + sem_medOff,
                    color=DEFAULT_COLORS['medOff'], alpha=0.3)

    # Labels
    ax.set_xlabel('Filtration Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Betti Number', fontsize=12, fontweight='bold')
    ax.set_title(title or f"Betti Curve Comparison (H{homology_dim})",
                fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"betti_comparison_h{homology_dim}.png",
                   dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def plot_heatmap_comparison(kernel_medOn: np.ndarray,
                           kernel_medOff: np.ndarray,
                           homology_dim: int = 1,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot side-by-side heatmaps comparing heat kernels for medOn vs medOff.

    Args:
        kernel_medOn: Heat kernel for medOn (n_dims, n, n) or (n, n)
        kernel_medOff: Heat kernel for medOff
        homology_dim: Which homology dimension to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Handle different input shapes
    if kernel_medOn.ndim == 4:
        kernel_medOn = kernel_medOn[0, homology_dim, :, :]
    elif kernel_medOn.ndim == 3:
        kernel_medOn = kernel_medOn[homology_dim, :, :]

    if kernel_medOff.ndim == 4:
        kernel_medOff = kernel_medOff[0, homology_dim, :, :]
    elif kernel_medOff.ndim == 3:
        kernel_medOff = kernel_medOff[homology_dim, :, :]

    # Create side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Determine common colorbar limits
    vmin = min(kernel_medOn.min(), kernel_medOff.min())
    vmax = max(kernel_medOn.max(), kernel_medOff.max())

    # MedOn heatmap
    im1 = axes[0].imshow(kernel_medOn, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('MedOn', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0])

    # MedOff heatmap
    im2 = axes[1].imshow(kernel_medOff, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('MedOff', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[1])

    # Difference heatmap
    difference = kernel_medOn - kernel_medOff
    diff_max = max(abs(difference.min()), abs(difference.max()))
    im3 = axes[2].imshow(difference, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (MedOn - MedOff)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Sample Index')
    plt.colorbar(im3, ax=axes[2])

    fig.suptitle(title or f"Heat Kernel Comparison (H{homology_dim})",
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"heatmap_comparison_h{homology_dim}.png",
                   dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


def plot_summary_panel(df: pd.DataFrame,
                      features: List[str],
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create multi-panel summary figure showing multiple features.

    Args:
        df: DataFrame from data_loader.load_all_patients()
        features: List of feature names to plot (max 6)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    n_features = min(len(features), 6)
    n_rows = (n_features + 1) // 2
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features[:n_features]):
        ax = axes[i]

        # Filter to paired patients
        patient_counts = df.groupby('patient_id')['med_state'].apply(lambda x: set(x))
        paired_patients = patient_counts[patient_counts.apply(
            lambda x: {'medOn', 'medOff'}.issubset(x))].index.tolist()
        df_paired = df[df['patient_id'].isin(paired_patients)].copy()

        # Create box plot
        sns.boxplot(data=df_paired, x='med_state', y=feature, hue='med_state',
                   palette={'medOn': DEFAULT_COLORS['medOn'],
                           'medOff': DEFAULT_COLORS['medOff']},
                   ax=ax, legend=False)

        # Overlay individual points
        sns.stripplot(data=df_paired, x='med_state', y=feature,
                     color='black', alpha=0.3, size=3, ax=ax)

        ax.set_title(feature, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Feature Comparison: MedOn vs MedOff', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "summary_panel.png", dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig


# Example usage and testing
if __name__ == "__main__":
    from data_loader import load_all_patients, load_patient_features, extract_array_features
    from analysis_utils import paired_ttest, create_summary_table

    print("="*70)
    print("VISUALIZATION UTILS MODULE - EXAMPLE USAGE")
    print("="*70)

    # Load data
    print("\nLoading patient data...")
    df = load_all_patients(verbose=False)
    print(f"Loaded {len(df)} rows from {df['patient_id'].nunique()} patients")

    # Example 1: Paired scatter plot
    print("\n\nExample 1: Creating paired scatter plot")
    print("-"*70)
    fig1 = plot_paired_scatter(df, 'h1_persistence_entropy',
                               title="H1 Persistence Entropy: MedOn vs MedOff")
    print("✓ Paired scatter plot created")
    plt.close(fig1)

    # Example 2: Distribution comparison
    print("\n\nExample 2: Creating violin plot")
    print("-"*70)
    fig2 = plot_distribution_comparison(df, 'h0_feature_count',
                                       plot_type='violin',
                                       group_by='hemisphere')
    print("✓ Violin plot created")
    plt.close(fig2)

    # Example 3: Forest plot
    print("\n\nExample 3: Creating forest plot")
    print("-"*70)
    features_to_test = ['h0_feature_count', 'h1_feature_count',
                       'h0_persistence_entropy', 'h1_persistence_entropy']
    results = [paired_ttest(df, f, verbose=False) for f in features_to_test]
    summary = create_summary_table(results)
    fig3 = plot_forest(summary)
    print("✓ Forest plot created")
    plt.close(fig3)

    # Example 4: Landscape comparison
    print("\n\nExample 4: Creating landscape comparison")
    print("-"*70)
    # Load landscapes for paired patients
    from data_loader import PATIENTS_PAIRED
    landscapes_medOn = []
    landscapes_medOff = []

    for patient in PATIENTS_PAIRED[:3]:  # Use first 3 for demo
        try:
            patient_data = load_patient_features(patient, verbose=False)
            landscapes = extract_array_features(patient_data, 'persistence_landscape')
            if 'medOn' in landscapes and 'hold' in landscapes['medOn']:
                landscapes_medOn.append(landscapes['medOn']['hold']['dominant'])
            if 'medOff' in landscapes and 'hold' in landscapes['medOff']:
                landscapes_medOff.append(landscapes['medOff']['hold']['dominant'])
        except:
            continue

    if landscapes_medOn and landscapes_medOff:
        landscapes_medOn = np.array(landscapes_medOn)
        landscapes_medOff = np.array(landscapes_medOff)
        fig4 = plot_landscape_comparison(landscapes_medOn, landscapes_medOff, homology_dim=1)
        print("✓ Landscape comparison created")
        plt.close(fig4)

    # Example 5: Summary panel
    print("\n\nExample 5: Creating summary panel")
    print("-"*70)
    fig5 = plot_summary_panel(df, features_to_test)
    print("✓ Summary panel created")
    plt.close(fig5)

    print("\n" + "="*70)
    print("✓ All visualization examples completed successfully!")
    print("="*70)
