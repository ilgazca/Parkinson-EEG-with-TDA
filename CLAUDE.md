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
- **EEG_TDA_Pipeline.py**: Python script version of the full pipeline (may be outdated)
- **EEG_TDA_Pipeline.ipynb**: Main analysis notebook with complete TDA workflow
- **Preprocess_for_TDA.ipynb**: Preprocessing-focused notebook (filtering, downsampling, slicing)
- **10_overview.ipynb**: MNE-Python tutorial (likely not project-specific)

### Data Organization
- Subject data stored in subdirectories (e.g., `i4oK0F/`)
- Raw .mat files in `sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/` (gitignored)
- Processed data saved as pickle files:
  - Time series slices: `medOff_left_hold.pkl`, `medOff_left_resting.pkl`, etc.
  - Persistence diagrams: `medOff_left_hold_diagrams.pkl`, etc.

## Common Workflow

### Running the Analysis
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

## Dependencies

Key libraries used:
- **numpy, pandas**: Data manipulation
- **scipy**: Signal processing (loadmat, decimate, butter, filtfilt)
- **matplotlib, plotly, seaborn**: Visualization
- **giotto-tda** (gtda): TDA computations
- **sklearn**: PCA, feature processing

## Notes for Development

- Always reload `eeg_utils` when making changes: `importlib.reload(eeg_utils)`
- Plotting uses light theme: `pio.templates.default = "plotly_white"` and `plt.style.use('default')`
- Most functions in `eeg_utils.py` have comprehensive docstrings and error handling
- Event times stored in subject directories as `.txt` files (e.g., `sub-i4oK0F event times.txt`)
