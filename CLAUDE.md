# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project analyzing Parkinson's disease EEG/LFP (Local Field Potential) data using Topological Data Analysis (TDA). The pipeline processes neural signals recorded during medication-on and medication-off states across different motor tasks (holding and resting states).

## Data Processing Pipeline

The analysis follows this workflow:
1. Load .mat files containing LFP/EEG data
2. Apply band-pass filtering (4-48 Hz)
3. Downsample from 2000 Hz to 100 Hz
4. Slice time series into task-specific windows (e.g., 60-second epochs)
5. Apply Takens embedding with automatic parameter search
6. Compute persistent homology using Vietoris-Rips persistence
7. Extract topological features (persistence entropy, landscapes, Betti curves, etc.)
8. **Aggregate multiple slices** into single representative features per condition
9. **Statistical analysis** to distinguish medOn vs medOff states

## Key Files

### Core Utilities
- **eeg_utils.py**: Main utility library containing all reusable functions
  - `mat_to_dataframe()`: Loads .mat files, auto-detects data/labels/sampling rate keys
  - `butter_bandpass_filter()`: Butterworth bandpass filtering
  - `downsample_eeg_dataframe()`: Downsamples using scipy's decimate
  - `fit_embedder()`: Fits Takens embedding with optimal parameter search
  - `filter_persistence_diagram()`: Filters features by lifespan threshold
  - `extract_features()`: Extracts summary statistics (count, lifespan, birth/death times) per homology dimension
  - `sample_time_series_slices()`: Systematic or arbitrary time series slicing
  - `pad_diagrams()`: Pads persistence diagrams for distance calculations

### Analysis Scripts & Notebooks
- **feature_extraction.py**: Automated TDA pipeline script for extracting features from LFP data
  - Processes .mat files and event times
  - Outputs persistence diagrams and all topological features
  - Usage: `python feature_extraction.py <mat_file> <events_file> <output_folder> --prefix medOff`
- **aggregate_features.py**: Aggregates multiple time slices into single representative features
  - Combines features from multiple events (e.g., 5 hold events → 1 representative value)
  - Supports mean, median, and full statistics aggregation
  - Usage: `python aggregate_features.py <patient_folder> --method mean`
- **batch_aggregate.py**: Batch processing for all patients
  - Runs aggregation on all patient folders automatically
  - Usage: `python batch_aggregate.py --method mean`
- **comparison_pipeline.ipynb**: **Main comparison notebook for analyzing aggregated features**
  - Part 1: Multi-patient analysis with summary statistics across all patients
  - Part 2: Two-patient comparison pipeline with flexible parameters
  - Compares medOn vs medOff states across dominant/non-dominant hemispheres
  - Handles both scalar features (entropy, counts, lifespans) and array features (landscapes, Betti curves, heat kernels)
  - Includes visualization capabilities and statistical comparisons
  - Usage: Set `patient1` and `patient2` variables to compare any two patients
- **example_load_aggregated.py**: Example script showing how to load and analyze aggregated features
- **EEG_TDA_Pipeline.py**: Python script version of the full pipeline (may be outdated)
- **EEG_TDA_Pipeline.ipynb**: Main analysis notebook with complete TDA workflow
- **Preprocess_for_TDA.ipynb**: Preprocessing-focused notebook (filtering, downsampling, slicing)
- **10_overview.ipynb**: MNE-Python tutorial (likely not project-specific)

### Documentation
- **ANALYSIS_METHODOLOGY.md**: Comprehensive guide to analysis pathways for distinguishing medOn vs medOff states
  - 10 analysis pathways (visual, statistical, distance-based, etc.)
  - Detailed implementation examples
  - Statistical testing approaches
- **AGGREGATION_GUIDE.md**: Complete guide to feature aggregation
  - Explains aggregation methods and when to use them
  - Workflow and examples
  - Troubleshooting tips

### Data Organization
- Subject data stored in subdirectories (e.g., `i4oK0F/`, `QZTsn6/`)
- Raw .mat files in `sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/` (gitignored)
- Processed data saved as pickle files per patient:
  - Time series slices: `medOff_left_holdL.pkl`, `medOff_left_resting.pkl`, etc.
  - Persistence diagrams: `medOff_left_holdL_diagrams.pkl`, `medOn_left_holdL_diagrams.pkl`, etc.
  - All features per medication state: `medOff_all_features_holdL.pkl`, `medOn_all_features_holdL.pkl`
  - **Aggregated features**: `aggregated_features_holdL.pkl` (combined from all slices)
- Event timing files: `event_times.txt` or `sub-{ID}_ses-PeriOp_task-*_events.tsv`

#### File Naming Convention
All feature files include a **hold indicator suffix** (`holdL` or `holdR`) that identifies which arm the subject raised during the experiment:

- **holdL**: Subject raised their LEFT arm (e.g., `medOff_left_holdL.pkl`, `aggregated_features_holdL.pkl`)
- **holdR**: Subject raised their RIGHT arm (e.g., `medOff_left_holdR.pkl`, `aggregated_features_holdR.pkl`)

This convention applies to:
- Time series slices: `{medState}_{hemisphere}_holdL.pkl` or `{medState}_{hemisphere}_holdR.pkl`
- Persistence diagrams: `{medState}_{hemisphere}_holdL_diagrams.pkl` or `{medState}_{hemisphere}_holdR_diagrams.pkl`
- All features: `{medState}_all_features_holdL.pkl` or `{medState}_all_features_holdR.pkl`
- Aggregated features: `aggregated_features_holdL.pkl` or `aggregated_features_holdR.pkl`
- Resting state files (no hold indicator): `{medState}_{hemisphere}_resting.pkl`

**Dominant vs Non-Dominant Hemisphere:**
Due to contralateral motor control (brain-body crossed connections):
- **holdL subjects**: RIGHT hemisphere is dominant (controls left arm)
  - Right channel = dominant/active hemisphere during hold task
  - Left channel = non-dominant/less active hemisphere
- **holdR subjects**: LEFT hemisphere is dominant (controls right arm)
  - Left channel = dominant/active hemisphere during hold task
  - Right channel = non-dominant/less active hemisphere

The aggregation scripts automatically map channels to `dominant` and `nondominant` labels based on the hold type.

## Common Workflow

### Complete Pipeline (Feature Extraction + Aggregation + Analysis)

#### Step 1: Extract Features from Raw Data
```bash
# Extract features for one patient
python feature_extraction.py \
    data/sub-i4oK0F_HoldL_MedOff.mat \
    i4oK0F/event_times.txt \
    ./i4oK0F/ \
    --prefix medOff

python feature_extraction.py \
    data/sub-i4oK0F_HoldL_MedOn.mat \
    i4oK0F/event_times.txt \
    ./i4oK0F/ \
    --prefix medOn
```

This creates:
- `i4oK0F/medOff_left_holdL_diagrams.pkl`
- `i4oK0F/medOff_all_features_holdL.pkl`
- `i4oK0F/medOn_all_features_holdL.pkl`

#### Step 2: Aggregate Multiple Slices
```bash
# Aggregate features for one patient
python aggregate_features.py ./i4oK0F/ --method mean

# Or batch process all patients
python batch_aggregate.py --method mean --include-variability
```

This creates:
- `i4oK0F/aggregated_features_holdL.pkl`

#### Step 3: Load and Analyze
```python
import pickle
import pandas as pd
from scipy import stats

# Load aggregated features
with open('i4oK0F/aggregated_features_holdL.pkl', 'rb') as f:
    data = pickle.load(f)

# Access features (now includes dominant/nondominant labels)
medOn_dominant_hold = data['medOn']['dominant_hold']  # Right hemisphere for holdL subjects
medOff_dominant_hold = data['medOff']['dominant_hold']
# Or use the original left/right labels
medOn_left_hold = data['medOn']['left_hold']
medOff_left_hold = data['medOff']['left_hold']

# Compare scalar features
print(f"MedOn entropy: {medOn_left_hold['persistence_entropy_mean']:.3f}")
print(f"MedOff entropy: {medOff_left_hold['persistence_entropy_mean']:.3f}")

# Handle array-based features (landscapes, Betti curves, heat kernels)
import numpy as np
persistence_landscape = medOn_dominant_hold['persistence_landscape_mean']
landscape_norm = np.linalg.norm(persistence_landscape)  # L2 norm for scalar summary
print(f"Persistence landscape L2 norm: {landscape_norm:.4f}")
```

#### Step 4: Use Comparison Pipeline (Recommended)
For comprehensive patient comparisons, use the `comparison_pipeline.ipynb` notebook:
1. Open `comparison_pipeline.ipynb`
2. Run cells 1-11 to load all patient data
3. Set `patient1` and `patient2` variables in cell 16 (e.g., `patient1 = 'i4oK0F'`, `patient2 = 'QZTsn6'`)
4. Run comparison cells (18-21) to compare specific conditions
5. Run visualization cells (22-23) for single patient medOn vs medOff analysis

The notebook includes:
- Multi-patient summary statistics
- Two-patient direct comparisons
- Single patient medOn vs medOff comparisons
- Flexible custom comparisons (any combination of patients, states, hemispheres)
- Proper handling of both scalar and array-based features

See **ANALYSIS_METHODOLOGY.md** for 10 detailed analysis pathways.

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

### Saving/Loading Results
```python
# Save
with open("path/to/output.pkl", "wb") as f:
    pickle.dump(data, f)

# Load
with open("path/to/output.pkl", "rb") as f:
    data = pickle.load(f)
```

## Technical Details

### Signal Processing Parameters
- **Original sampling rate**: 2000 Hz
- **Target sampling rate**: 100 Hz (after decimation)
- **Band-pass filter**: 4-48 Hz (Butterworth, order 5)
- **Decimation method**: FIR filter with zero-phase

### TDA Parameters
- **Takens embedding**:
  - Max dimension: 20-30
  - Max time delay: 1000 samples (10 seconds at 100 Hz)
  - Search mode finds optimal parameters automatically
- **Homology dimensions**: H0, H1, H2, H3
- **Typical embedding subset size**: 250 points (to manage computational cost)
- **Lifespan threshold**: ~0.2 (for filtering trivial topological features)

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

### Distance Metrics
When computing pairwise distances between diagrams:
1. Pad diagrams using `pad_diagrams()` to equalize dimensions
2. Scale using `Scaler(metric='wasserstein')` or other metrics
3. Compute distances using `PairwiseDistance(metric='wasserstein')` or `'bottleneck'`

### Aggregated Feature Types
Aggregated features come in two types:

#### Scalar Features (directly comparable)
These are single numerical values suitable for tabular comparison and standard visualizations:
- `persistence_entropy_mean`, `persistence_entropy_std`
- `h0-h3_feature_count_mean`, `h0-h3_feature_count_std`
- `h0-h3_avg_lifespan_mean`, `h0-h3_avg_lifespan_std`
- `h0-h3_max_lifespan_mean`, `h0-h3_max_lifespan_std`
- `h0-h3_std_lifespan_mean`, `h0-h3_std_lifespan_std`
- `h1-h3_avg_birth_mean`, `h1-h3_avg_death_mean` (and std versions)

#### Array-Based Features (require special handling)
These are numpy arrays representing full functional/vectorized representations:
- `persistence_landscape_mean`, `persistence_landscape_std` - Shape: (n_layers, n_points) or (1, n_layers, n_points)
- `betti_curve_mean`, `betti_curve_std` - Shape: (n_dimensions, n_filtration_values)
- `heat_kernel_mean`, `heat_kernel_std` - Shape: (n_dimensions, n_bins, n_bins)

**Important**: Array features cannot be directly included in comparison tables. Use these approaches:
- **L2 norm**: Compute `np.linalg.norm(array)` for scalar summary
- **Distance metrics**: Use Wasserstein or Bottleneck distance for comparisons
- **Visualization**: Use heatmaps, line plots, or contour plots to display
- **Vectorization**: Flatten arrays for machine learning pipelines

The `comparison_pipeline.ipynb` notebook handles both types automatically, showing scalar features in tables and array features as L2 norms in separate sections.

## Dependencies

Key libraries used:
- **numpy, pandas**: Data manipulation
- **scipy**: Signal processing (loadmat, decimate, butter, filtfilt)
- **matplotlib, plotly, seaborn**: Visualization
- **giotto-tda** (gtda): TDA computations
- **sklearn**: PCA, feature processing

## Project Goals

The main objective is to **distinguish between medOn and medOff states** in Parkinson's disease patients using topological features from LFP/MEG recordings, **without machine learning**. This involves:

1. Extracting topological features from neural signals (completed for some patients)
2. Aggregating multiple event slices into representative features (scripts created, in progress)
3. Direct statistical comparison and manual analysis (next step)

### Research Questions
- Can topological features distinguish medication states?
- Which features (H0, H1, H2, H3) are most discriminative?
- Are effects lateralized (left vs right hemisphere)?
- Do effects differ between resting and active (hold) states?

### Current Status (as of latest update)
- ✅ Feature extraction pipeline: `feature_extraction.py` (complete)
- ✅ Aggregation scripts: `aggregate_features.py`, `batch_aggregate.py` (complete)
- ✅ Analysis methodology: 10 pathways documented in `ANALYSIS_METHODOLOGY.md`
- ✅ Comparison pipeline: `comparison_pipeline.ipynb` (complete) - allows flexible patient comparisons
- 🔄 Feature extraction: Completed for 6 patients (0cGdk9, 2IU8mi, 2IhVOz, i4oK0F, jyC0j3, QZTsn6)
- 🔄 Feature aggregation: Completed for 6 patients with full metadata
- 🔄 Statistical analysis: In progress using comparison pipeline (see ANALYSIS_METHODOLOGY.md)

### Dataset
- **23 patients** total
- **2 medication states** per patient: medOn, medOff
- **2 hemispheres**: left, right
- **2 conditions**: resting, hold task
- **~5 event slices** per condition

## Notes for Development

- Always reload `eeg_utils` when making changes: `importlib.reload(eeg_utils)`
- Plotting uses light theme: `pio.templates.default = "plotly_white"` and `plt.style.use('default')`
- Most functions in `eeg_utils.py` have comprehensive docstrings and error handling
- Event times stored in subject directories as `.txt` files (e.g., `event_times.txt`)
- For aggregation: use `--method mean` for standard analysis, `--method full --include-variability` for comprehensive analysis
- Always push .pkl files when doing a git push
- All the scripts and notebooks use the conda environment named `TDA`. Whenever you need to run tests on a script, always check if the environment is activated, and activate it if not
- Never ask permission to edit @CLAUDE.md file