# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project analyzing Parkinson's disease EEG/LFP (Local Field Potential) data using Topological Data Analysis (TDA). The pipeline processes neural signals recorded during medication-on and medication-off states across different motor tasks (holding and resting states).

## Current Project Status

**This project has been reset to a clean starting point.** All intermediate data files (.pkl files) have been deleted. We are starting fresh with:
- Raw .mat data files (in `sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/`)
- Extracted event times for all patients (in each patient folder as `event_times.txt`)
- Core utility functions (`eeg_utils.py`)
- Feature extraction script (`feature_extraction.py`)
- Main analysis notebook (`EEG_TDA_Pipeline.ipynb`)

## Data Processing Pipeline

The planned workflow is:
1. Load .mat files containing LFP/EEG data
2. Apply band-pass filtering (4-48 Hz)
3. Downsample from 2000 Hz to 100 Hz
4. Slice time series into task-specific windows using extracted event times
5. Apply Takens embedding with automatic parameter search
6. Compute persistent homology using Vietoris-Rips persistence
7. Extract topological features (persistence entropy, landscapes, Betti curves, etc.)
8. **[Future]** Aggregate and analyze features to distinguish medOn vs medOff states

## Key Files

### Core Utilities

#### Signal Processing & TDA
- **eeg_utils.py**: Main utility library for signal processing and TDA
  - `mat_to_dataframe()`: Loads .mat files, auto-detects data/labels/sampling rate keys
  - `butter_bandpass_filter()`: Butterworth bandpass filtering
  - `downsample_eeg_dataframe()`: Downsamples using scipy's decimate
  - `fit_embedder()`: Fits Takens embedding with optimal parameter search
  - `filter_persistence_diagram()`: Filters features by lifespan threshold
  - `extract_features()`: Extracts summary statistics (count, lifespan, birth/death times) per homology dimension
  - `sample_time_series_slices()`: Systematic or arbitrary time series slicing
  - `pad_diagrams()`: Pads persistence diagrams for distance calculations

#### Analysis Infrastructure (NEW)
- **data_loader.py**: Data loading and organization utilities
  - `load_all_patients()`: Load all 14 patients into unified pandas DataFrame (92 rows, 28 scalar features)
  - `load_patient_features()`: Load single patient's feature files with verbose output
  - `extract_scalar_features()`: Extract scalar features into DataFrame format
  - `extract_array_features()`: Extract landscapes, Betti curves, or heat kernels
  - `get_paired_patients_data()`: Filter to patients with both medOn and medOff (9 patients)
  - `load_persistence_diagrams()`: Load raw persistence diagrams
  - Uses clean naming: 'dominant'/'nondominant' hemispheres, 'hold'/'resting' conditions

- **analysis_utils.py**: Statistical analysis and testing
  - `paired_ttest()`: Paired t-test with Cohen's d effect size and 95% CI
  - `wilcoxon_test()`: Non-parametric paired test (Wilcoxon signed-rank)
  - `independent_ttest()`: Independent samples t-test for group comparisons
  - `compute_effect_size()`: Cohen's d calculation with interpretation
  - `multiple_comparison_correction()`: FDR (Benjamini-Hochberg) and Bonferroni correction
  - `create_summary_table()`: Generate results summary with optional p-value correction
  - Array feature summarization:
    - `summarize_persistence_landscape()`: 32 metrics (L1/L2/Linf norms, AUC, peaks)
    - `summarize_betti_curve()`: 24 metrics (AUC, max Betti, centroid)
    - `summarize_heat_kernel()`: 24 metrics (Frobenius norm, trace, statistics)

- **visualization_utils.py**: Plotting and visualization functions
  - `plot_paired_scatter()`: Paired scatter plots (medOn vs medOff per patient)
  - `plot_distribution_comparison()`: Box/violin plots for feature distributions
  - `plot_forest()`: Forest plots with effect sizes and confidence intervals
  - `plot_landscape_comparison()`: Mean persistence landscapes with SEM error bands
  - `plot_betti_comparison()`: Mean Betti curves with SEM error bands
  - `plot_heatmap_comparison()`: Side-by-side heat kernel heatmaps with difference maps
  - `plot_summary_panel()`: Multi-panel figures showing multiple features
  - All functions support optional grouping (hemisphere/condition) and auto-save

### Analysis Scripts & Notebooks
- **batch_feature_extraction.py**: **NEW** Simplified batch processing script for all patients
  - Processes all 14 patients automatically
  - Extracts ONE 5-second slice from resting and ONE from first hold event
  - Avoids edge artifacts by centering slices within events
  - Handles patients with only medOn or only medOff data
  - Saves features in patient directories with proper naming
  - Usage: `python batch_feature_extraction.py`
- **feature_extraction.py**: Original single-patient TDA pipeline script
  - Processes .mat files and event times for one patient at a time
  - Supports multiple slices and customizable parameters
  - Usage: `python feature_extraction.py <mat_file> <events_file> <output_folder> --med-state medOff`
- **event_time_extractor.py**: Helper script for extracting event times from .tsv files
  - Processes BIDS-formatted event files
  - Outputs formatted event times showing trial type, onset, duration, and end times
- **EEG_TDA_Pipeline.ipynb**: Main analysis notebook with complete TDA workflow
  - Interactive notebook for exploring the full pipeline
  - Includes preprocessing, TDA computation, and feature extraction
- **01_Exploratory_Analysis.ipynb**: Exploratory data analysis notebook (COMPLETED)
  - Comprehensive distribution analysis of topological features
  - **Rigorous normality assessment** with medication state separation:
    - Shapiro-Wilk tests on medOn, medOff, and paired differences (Cell 8 - pooled)
    - Hemisphere-specific normality tests (Cell 21 - dominant/nondominant)
    - Proper statistical approach: tests paired differences (key assumption for paired t-tests)
  - **Automated test selection**: Cells automatically read normality results and select appropriate tests
    - Cell 26 (pooled tests): Reads from Cell 8
    - Cells 31-32 (hemisphere tests): Read from Cell 21
  - **MedOn vs MedOff visual comparisons**:
    - Overlapping histograms showing medication effects (Cells 5, 14)
    - Separate statistics tables comparing states
    - Hemisphere-specific visualizations
  - **Statistical testing with FDR correction**:
    - Pooled analysis (12 features): Cell 26-27
    - Hemisphere-specific analysis (24 tests): Cells 31-33
    - Shows both uncorrected and FDR-corrected significance
  - Quantitative lateralization comparison with lateralization index (Cell 34)

### Documentation
- **NAMING_CONVENTION_UPDATE.md**: File naming conventions
  - Explains holdL vs holdR naming
  - Describes dominant vs non-dominant hemisphere mapping
- **Analysis_Pipeline.md**: Comprehensive statistical analysis methodology
  - Detailed comparison strategies for each feature type (scalars, landscapes, Betti curves, heat kernels)
  - Multi-level analysis approach (within-patient, group-level, hemisphere, condition)
  - 5-phase workflow from EDA to final summary
  - Scaling guidelines by feature type
  - Code structure templates and implementation guide
  - Expected outcomes and reporting standards
- **ToDo.md**: Project task checklist
  - Tracks implementation of analysis infrastructure
  - Organized into 5 phases (Core Utilities, Directory Structure, Configuration, Notebooks, Automation)
  - Checkboxes for tracking progress
- **results/README.md**: Results directory documentation
  - Explains purpose of each subdirectory
  - File naming conventions for outputs
  - Usage examples and maintenance guidelines

### Data Organization
- **Patient folders**: 14 subdirectories named by patient ID (e.g., `i4oK0F/`, `QZTsn6/`, `AB2PeX/`, etc.)
  - Each contains `event_times.txt` with extracted event timing information
  - Contains processed .pkl files with extracted features:
    - Persistence diagrams: `{medState}_{dominant/nondominant}_{condition}_{holdType}_diagrams.pkl`
    - All features: `{medState}_all_features_{holdType}.pkl`
    - Visualization plots: `{medState}_{hemisphere}_{condition}_*.png`

- **Raw .mat files**: Located in `sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/` directory
  - Contains LFP data for all patients
  - Format: `sub-{ID}_{HoldL/HoldR}_{MedOn/MedOff}_*_LFP_Hilbert.mat`

- **Patient data availability**: See `patient_data_availability.tsv`
  - Lists which patients have medOn and/or medOff data
  - 9 patients with both states (PAIRED - most useful for analysis)
  - 3 with only medOff (AB2PeX, gNX5yb, PuPVlx)
  - 2 with only medOn (FYbcap, VopvKx)

- **Results directory** (NEW): Organized output structure
  - `results/exploratory/`: EDA outputs, distributions, correlations, PCA
  - `results/statistical_tests/`: Test results, summary tables (CSV/Excel)
  - `results/figures/`: Publication-quality figures
    - `scalar_features/`: Paired scatter plots, box/violin plots, forest plots
    - `landscapes/`: Landscape comparisons with error bands
    - `betti_curves/`: Betti curve comparisons
    - `heat_kernels/`: Heatmaps and difference maps
    - `summary/`: Multi-panel figures
  - `results/reports/`: Analysis reports and summaries

#### Patient List
Currently have event times extracted for 14 patients:
- 0cGdk9, 2IhVOz, 2IU8mi, AB2PeX, AbzsOg, BYJoWR, dCsWjQ
- FYbcap, gNX5yb, i4oK0F, jyC0j3, PuPVlx, QZTsn6, VopvKx

#### File Naming Convention
Data files follow these conventions:
- **holdL**: Subject raised their LEFT arm during the task
- **holdR**: Subject raised their RIGHT arm during the task
- Raw data files: `sub-{ID}_{HoldL/HoldR}_{MedOn/MedOff}_*_LFP_Hilbert.mat`
- Event files: `event_times.txt` (formatted output from .tsv event files)

**Dominant vs Non-Dominant Hemisphere:**
Due to contralateral motor control (brain controls opposite side of body):
- **holdL subjects**: RIGHT hemisphere is dominant (controls left arm)
- **holdR subjects**: LEFT hemisphere is dominant (controls right arm)

## Common Workflow

### Simplified Batch Processing (Recommended)

**Quick Start:**
```bash
# Activate conda environment
conda activate TDA

# Run batch processing on all patients
python batch_feature_extraction.py
```

This will:
1. Process all 14 patients automatically
2. For each patient, extract features from available .mat files (medOn and/or medOff)
3. Extract ONE 5-second slice from resting period (centered, avoiding edges)
4. Extract ONE 5-second slice from first hold event (centered, avoiding edges)
5. Compute Takens embeddings
6. Generate persistence diagrams
7. Extract all topological features:
   - Persistence Entropy
   - Summary statistics (H0-H3 feature counts, lifespans, birth/death times)
   - Persistence Landscapes
   - Betti Curves
   - Heat Kernel signatures
8. Save results in each patient's directory with proper naming

**Output files per patient:**
- `{medState}_{hemisphere}_hold_{holdType}_diagrams.pkl` - Persistence diagrams for hold task
- `{medState}_{hemisphere}_resting_diagrams.pkl` - Persistence diagrams for resting
- `{medState}_all_features_{holdType}.pkl` - All extracted features

**Example outputs for patient i4oK0F (holdL subject with both medOn and medOff):**
- `medOff_left_hold_holdL_diagrams.pkl`
- `medOff_right_hold_holdL_diagrams.pkl`
- `medOff_left_resting_diagrams.pkl`
- `medOff_right_resting_diagrams.pkl`
- `medOff_all_features_holdL.pkl`
- `medOn_left_hold_holdL_diagrams.pkl`
- `medOn_right_hold_holdL_diagrams.pkl`
- `medOn_left_resting_diagrams.pkl`
- `medOn_right_resting_diagrams.pkl`
- `medOn_all_features_holdL.pkl`

### Alternative: Single Patient Processing

For more control or testing, use the original script:
```bash
python feature_extraction.py \
    sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/sub-i4oK0F_HoldL_MedOff_*.mat \
    i4oK0F/event_times.txt \
    ./i4oK0F/ \
    --prefix medOff \
    --slice-length 5
```

### Analysis Workflow (Recommended)

**Quick Start for Statistical Analysis:**
```python
from data_loader import load_all_patients, get_paired_patients_data
from analysis_utils import paired_ttest, create_summary_table, multiple_comparison_correction
from visualization_utils import plot_paired_scatter, plot_forest, plot_distribution_comparison

# Load all patient data
df = load_all_patients(verbose=True)

# Get only paired patients (9 patients with both medOn and medOff)
df_paired = get_paired_patients_data(df)

# Perform statistical tests
features = ['h0_feature_count', 'h1_feature_count', 'h0_persistence_entropy', 'h1_persistence_entropy']
results = [paired_ttest(df, feature, verbose=True) for feature in features]

# Create summary table with FDR correction
summary = create_summary_table(results, include_corrected=True)
print(summary)

# Visualize results
fig1 = plot_paired_scatter(df, 'h1_persistence_entropy',
                           save_path='results/figures/scalar_features/')
fig2 = plot_forest(summary, save_path='results/figures/summary/')
fig3 = plot_distribution_comparison(df, 'h0_feature_count', plot_type='violin',
                                   save_path='results/figures/scalar_features/')
```

### Interactive Exploration

Use `EEG_TDA_Pipeline.ipynb` for:
- Visualizing raw LFP data
- Testing preprocessing parameters
- Experimenting with TDA settings
- Visualizing persistence diagrams

### Running the Analysis (Manual/Notebook)
```python
# Standard imports for TDA analysis
from eeg_utils import *
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, PersistenceLandscape, BettiCurve
import pickle

# Load and preprocess
df, left_lfp, right_lfp, left_name, right_name = mat_to_dataframe("path/to/file.mat")
left_filtered = butter_bandpass_filter(left_lfp, lowcut=4, highcut=48, fs=2000, order=5)
df_downsampled = downsample_eeg_dataframe(df, original_fs=2000, target_fs=100)

# Slice time series (for task-specific epochs)
slices = sample_time_series_slices(time_series, slice_length=60.0, n_slices=5)

# Takens embedding
embedder = SingleTakensEmbedding(parameters_type="search", time_delay=1000, dimension=20, n_jobs=-1)
embedding = fit_embedder(embedder, time_series_slice)

# Persistent homology
persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2, 3], n_jobs=-1)
diagram = persistence.fit_transform(embedding[None, :, :])

# Feature extraction
features = extract_features(diagram, homology_dimensions=[0, 1, 2, 3])
```

### Saving/Loading Results (For Future Use)
```python
import pickle

# Save processed data
with open("path/to/output.pkl", "wb") as f:
    pickle.dump(data, f)

# Load processed data
with open("path/to/output.pkl", "rb") as f:
    data = pickle.load(f)
```

## Technical Details

### Signal Processing Parameters
- **Original sampling rate**: 2000 Hz
- **Target sampling rate**: 100 Hz (after decimation)
- **Band-pass filter**: 4-48 Hz (Butterworth, order 5)
- **Decimation method**: FIR filter with zero-phase
- **Slice length**: 5 seconds (simplified approach, was 60 seconds before)
- **Slices per condition**: 1 (one from resting, one from first hold event)
- **Edge avoidance**: 1-second buffer from event boundaries

### TDA Parameters
- **Takens embedding**:
  - Max dimension: 20
  - Max time delay: 1000 samples (10 seconds at 100 Hz)
  - Search mode finds optimal parameters automatically
- **Homology dimensions**: H0, H1, H2, H3
- **Embedding subset size**: 250 points (to manage computational cost)
- **Features extracted**:
  - Persistence Entropy
  - Summary statistics (feature count, avg/max/std lifespan, avg birth/death per dimension)
  - Persistence Landscapes
  - Betti Curves
  - Heat Kernel signatures

### Channel Naming Convention
LFP channels follow pattern: `LFP-{side}-{number}` (e.g., "LFP-left-78", "LFP-right-23")

## Data Structure Notes

### .mat File Format
- Data matrix key: Auto-detected (looks for 2D array with channels × samples)
- Labels key: Tries 'target_labels', 'labels', 'channel_labels', 'ch_names'
- Sampling rate key: Tries 'fs', 'Fs', 'fsample', 'sampling_rate', 'srate'
- Default to 2000 Hz if sampling rate not found

### Persistence Diagrams
- Shape: `(n_samples, n_features, 3)` where columns are `[birth, death, dimension]`
- Infinite death times represented as `np.inf`
- Features with birth == death are trivial and typically filtered out

### Distance Metrics (For Future Analysis)
When computing pairwise distances between diagrams:
1. Pad diagrams using `pad_diagrams()` to equalize dimensions
2. Scale using `Scaler(metric='wasserstein')` or other metrics
3. Compute distances using `PairwiseDistance(metric='wasserstein')` or `'bottleneck'`

### Feature Types
TDA features extracted from persistence diagrams include:

#### Scalar Features
Single numerical values per homology dimension:
- Feature counts (number of topological features)
- Lifespan statistics (mean, max, std)
- Birth/death time statistics
- Persistence entropy

#### Array-Based Features
Full vectorized representations:
- Persistence landscapes
- Betti curves
- Heat kernels
- Persistence images

## Dependencies

Key libraries used:
- **numpy, pandas**: Data manipulation
- **scipy**: Signal processing (loadmat, decimate, butter, filtfilt)
- **matplotlib, plotly, seaborn**: Visualization
- **giotto-tda** (gtda): TDA computations
- **sklearn**: PCA, feature processing

## Project Goals

The main objective is to **distinguish between medOn and medOff states** in Parkinson's disease patients using topological features from LFP recordings. This involves:

1. **Extract topological features** from neural signals during different tasks and medication states
2. **Analyze and compare** features to identify patterns distinguishing medOn vs medOff
3. **Understand lateralization** effects (left vs right hemisphere differences)

### Research Questions
- Can topological features distinguish medication states?
- Which features (H0, H1, H2, H3) are most discriminative?
- Are effects lateralized (left vs right hemisphere)?
- Do effects differ between resting and active (hold) states?

### Current Status

**Analysis Infrastructure Complete - Ready for Statistical Comparison**

#### Completed
- ✅ Event times extracted for 14 patients
- ✅ Patient data availability documented (`patient_data_availability.tsv`)
- ✅ Core signal processing utilities (`eeg_utils.py`)
- ✅ Batch feature extraction script (`batch_feature_extraction.py`)
- ✅ Feature extraction completed for all patients
- ✅ **Analysis infrastructure built:**
  - `data_loader.py`: Load and organize all patient features
  - `analysis_utils.py`: Statistical tests, effect sizes, summarization
  - `visualization_utils.py`: Publication-quality plots
- ✅ **Results directory structure** created and documented
- ✅ **Comprehensive analysis guide** (`Analysis_Pipeline.md`)

#### Data Summary
- **14 patients** with extracted features (92 total observations)
- **9 paired patients** with both medOn and medOff (primary analysis group)
- **28 scalar features** per observation (7 per homology dimension H0-H3)
- **Array features** available: landscapes, Betti curves, heat kernels
- **2 hemispheres** × **2 conditions** × **2 medication states** = rich multi-level data

#### Analysis Capabilities
- Paired t-tests with Cohen's d effect size
- Non-parametric alternatives (Wilcoxon)
- Multiple comparison correction (FDR, Bonferroni)
- Grouped analyses (by hemisphere, condition)
- Array feature summarization (landscapes → 32 metrics, Betti curves → 24 metrics)
- Comprehensive visualizations (scatter, violin, forest, landscapes, heatmaps)

**Completed:**
1. **Exploratory Data Analysis** - **COMPLETED**
   - ✅ Created `01_Exploratory_Analysis.ipynb`
   - ✅ Load all features and examine distributions
   - ✅ Medication state-specific distributions (medOn vs medOff overlapping histograms)
   - ✅ **Rigorous normality assessment**:
     - Pooled analysis: Tests medOn, medOff, and paired differences (Cell 8)
     - Hemisphere-specific: Tests medOn, medOff, and differences per hemisphere (Cell 21)
     - Proper criterion: Based on paired difference normality for test selection
   - ✅ Hemisphere-specific distribution analysis (dominant vs nondominant)
   - ✅ Quantitative lateralization comparison

2. **Statistical Testing** - **COMPLETED**
   - ✅ **Pooled analysis** (Cell 26-27):
     - Automatic test selection based on Cell 8 normality results
     - Paired t-tests for features with normal differences
     - Wilcoxon signed-rank tests for features with non-normal differences
     - FDR correction (Benjamini-Hochberg) across 12 tests
     - Shows both uncorrected and corrected significance
   - ✅ **Hemisphere-specific analysis** (Cells 31-33):
     - Automatic test selection based on Cell 21 normality results
     - Separate analysis for dominant and nondominant hemispheres
     - FDR correction across 24 tests (12 features × 2 hemispheres)
     - Lateralization comparison
     - Shows both uncorrected and corrected significance

**Key Findings:**
- **Pooled analysis**: 5/12 features significant before FDR correction, 0/12 after correction
- **Hemisphere-specific**: 4/24 tests significant before correction (all in nondominant hemisphere), 0/24 after correction
- **Lateralization pattern detected**: Medication effects stronger in nondominant hemisphere
- **Strongest effects**: H3 entropy, H1 entropy, H1 feature count (all nondominant hemisphere)

**Next Steps:**
1. **Extended Analysis** (Optional)
   - Outlier detection and sensitivity analysis
   - Correlation analysis between features
   - PCA/dimensionality reduction visualization
   - Condition-specific effects (resting vs hold)
2. **Manuscript Preparation**
   - Methods section (statistical parameters documented)
   - Results tables (LaTeX code generated)
   - Discussion of null findings and lateralization patterns
   - Power analysis for sample size justification

### Dataset
- **14 patients** currently prepared (with event times)
- **2 medication states** per patient: medOn, medOff
- **2 hemispheres**: left, right (LFP channels)
- **Multiple events** per recording: rest periods, hold tasks, bad segments
- **Raw data format**: .mat files with LFP time series

## Notes for Development

### Environment & Dependencies
- **Environment**: All scripts and notebooks use the conda environment named `TDA`
  - Always check if environment is activated before running scripts
  - Activate with: `conda activate TDA`
- **Key dependencies**: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, statsmodels, giotto-tda

### Module Usage
- **data_loader.py**: Load patient data before analysis
  - Use `load_all_patients(verbose=True)` to see data structure
  - Use `get_paired_patients_data()` to filter to paired patients
  - Clean naming convention: 'dominant'/'nondominant', 'hold'/'resting'
- **analysis_utils.py**: Statistical testing and summarization
  - Use `paired_ttest()` for primary within-patient comparisons
  - Use `create_summary_table()` with `include_corrected=True` for multiple comparison correction
  - Use `summarize_*()` functions to convert array features to scalars
- **visualization_utils.py**: Automatic figure generation
  - Always provide `save_path` parameter to save figures
  - Figures save to `results/figures/` subdirectories
  - Use `verbose=True` in analysis functions before plotting

### Code Organization
- **Utilities**: Reload modules when making changes: `importlib.reload(module_name)`
- **Plotting**: Uses light theme (`plt.style.use('default')`)
- **Event times**: Stored in each patient's directory as `event_times.txt`
- **Data files**:
  - Raw .mat files: `sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/`
  - Processed .pkl files: Individual patient folders
  - Analysis outputs: `results/` directory structure
- **Git workflow**:
  - .pkl files in patient folders: Not committed (too large)
  - Analysis outputs in results/: Decision pending
  - .md files: Can be edited without asking permission

### Analysis Workflow Best Practices
- **Always start with paired comparisons**: 9 paired patients provide most statistical power
- **Test normality of paired differences, not individual groups**: For paired t-tests, the critical assumption is normality of the differences (medOn - medOff), not the individual medication states. This is implemented in Cells 8 and 21.
- **Automatic test selection**: Configure cells to read normality results and automatically apply appropriate tests (t-test vs Wilcoxon) based on difference normality
- **Use effect sizes**: Don't rely on p-values alone; Cohen's d provides magnitude
- **Apply multiple comparison correction**: Use FDR (Benjamini-Hochberg, less conservative) across entire test family
  - Pooled analysis: 12 tests (one per feature)
  - Hemisphere-specific: 24 tests (12 features × 2 hemispheres)
- **Show both uncorrected and corrected results**: Transparency about correction impact (implemented in Cells 27 and 33)
- **Visualize medication states separately**: Use overlapping histograms to show medOn vs medOff distributions before statistical testing
- **Hemisphere-specific analysis**: Test each hemisphere separately - lateralization may reveal important patterns even when pooled analysis shows null results
- **Save systematically**: Use consistent naming and save to appropriate results/ subdirectories
- **Document findings**: Update results/reports/ with analysis summaries

## Important Reminders

- **Take it slow**: Careful, incremental analysis with proper validation
- **Verify assumptions**: Check normality, equal variance before parametric tests
- **Test on subsets first**: Try analysis on a few features before running all
- **Document decisions**: Keep track of parameter choices and their rationale in notebooks/reports
- **Use verbose output**: Enable verbose mode to understand what functions are doing
- **Check data quality**: Always examine loaded data structure before analysis
- Do not ask permission for creating and editing .md files.
- Don't ask for permission for running small python scripts that you use for reading the notebook