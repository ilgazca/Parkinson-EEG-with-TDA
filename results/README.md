# Results Directory

This directory contains all analysis outputs, figures, and reports from the Parkinson's disease TDA analysis comparing medOn vs medOff medication states.

## Directory Structure

### `exploratory/`
Exploratory Data Analysis (EDA) outputs:
- Distribution plots and histograms
- Correlation matrices
- PCA visualizations
- Initial data exploration notebooks outputs
- Outlier detection results

### `statistical_tests/`
Statistical test results and summary tables:
- Paired t-test results (CSV/Excel)
- Effect size calculations
- Multiple comparison correction results
- ANOVA and mixed-effects model outputs
- Summary tables (significant findings)

### `figures/`
Publication-quality figures organized by feature type:

#### `figures/scalar_features/`
Visualizations for scalar features:
- Paired scatter plots
- Box/violin plots for feature counts
- Forest plots for lifespan statistics
- Persistence entropy comparisons

#### `figures/landscapes/`
Persistence landscape visualizations:
- Mean landscape curves (medOn vs medOff)
- Error band plots (SEM)
- Landscape difference plots
- Per-homology dimension comparisons (H0-H3)

#### `figures/betti_curves/`
Betti curve visualizations:
- Mean Betti curves with error bands
- Filtration value comparisons
- Per-homology dimension analyses
- AUC and peak comparisons

#### `figures/heat_kernels/`
Heat kernel signature visualizations:
- Side-by-side heatmaps
- Difference maps (medOn - medOff)
- Spectral analyses
- Per-homology dimension heatmaps

#### `figures/summary/`
Comprehensive multi-panel figures:
- Top discriminative features
- Forest plots with all significant results
- Multi-feature summary panels
- Hemisphere and condition comparisons

### `reports/`
Analysis reports and summaries:
- HTML/PDF reports
- Comprehensive analysis summaries
- Results narratives
- Tables of significant findings
- Publication-ready summaries

## Usage

Results are automatically saved to these directories by the analysis scripts and notebooks when using the `save_path` parameter in visualization functions.

Example:
```python
from visualization_utils import plot_paired_scatter

fig = plot_paired_scatter(df, 'h1_persistence_entropy',
                          save_path='results/figures/scalar_features/')
```

## File Naming Convention

Files are named descriptively to indicate:
- Analysis type (e.g., `paired_scatter`, `forest_plot`, `landscape_comparison`)
- Feature name (e.g., `h0_feature_count`, `h1_persistence_entropy`)
- Grouping if applicable (e.g., `by_hemisphere`, `by_condition`)
- File format (`.png` for figures, `.csv` for tables, `.html` for reports)

Examples:
- `paired_scatter_h1_persistence_entropy_by_hemisphere.png`
- `forest_plot.png`
- `landscape_comparison_h1.png`
- `summary_table_significant_features.csv`

## Maintenance

- Keep directory structure organized
- Remove obsolete or duplicate files periodically
- Document major analysis runs in reports/
- Use version control for tracking important results
