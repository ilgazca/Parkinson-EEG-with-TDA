# Feature Aggregation Guide

This guide explains how to use the aggregation scripts to combine multiple time slices into single representative features.

## Overview

The feature extraction pipeline (`feature_extraction.py`) produces multiple slices per condition (e.g., 5 hold events and 5 resting events). Before analysis, we need to aggregate these into single feature vectors.

## Scripts

### 1. `aggregate_features.py` - Single Patient Aggregation

Aggregates features for a single patient.

#### Basic Usage

```bash
# Default: mean aggregation
python aggregate_features.py ./i4oK0F/

# Robust aggregation using median
python aggregate_features.py ./i4oK0F/ --method median

# Full statistics (mean, std, min, max, median, range, IQR)
python aggregate_features.py ./i4oK0F/ --method full

# Include variability features (std, range, IQR, CV)
python aggregate_features.py ./i4oK0F/ --method mean --include-variability

# Use bottleneck distance for diagrams instead of Wasserstein
python aggregate_features.py ./i4oK0F/ --distance-metric bottleneck

# Custom output filename
python aggregate_features.py ./i4oK0F/ --output my_custom_name
```

#### Aggregation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `mean` | Simple mean and std | Default, fast, works for most cases |
| `median` | Median and MAD (median absolute deviation) | Robust to outliers |
| `full` | Mean, std, min, max, median, range, IQR | Maximum information, larger feature set |

#### Output

Creates `aggregated_features.pkl` (or custom name) in the patient folder with structure:

```python
{
    'medOn': {
        'left_hold': {
            'hemisphere': 'left',
            'condition': 'hold',
            'persistence_entropy_mean': 2.34,
            'persistence_entropy_std': 0.12,
            'H0_count_mean': 15.2,
            'H0_count_std': 2.1,
            'persistence_diagram': array(...),  # Representative diagram
            'medoid_index': 2,  # Which slice was selected as representative
            'betti_curve_mean': array(...),
            'betti_curve_std': array(...),
            # ... more features
        },
        'left_resting': { ... },
        'right_hold': { ... },
        'right_resting': { ... },
        'left_wasserstein': array(...),  # Distance matrix (if computed)
        'right_wasserstein': array(...),
    },
    'medOff': {
        'left_hold': { ... },
        # ... same structure
    }
}
```

---

### 2. `batch_aggregate.py` - Batch Processing for All Patients

Runs aggregation on all patient folders automatically.

#### Basic Usage

```bash
# Aggregate all patients in current directory
python batch_aggregate.py

# Full statistics with variability for all patients
python batch_aggregate.py --method full --include-variability

# Process only specific patients
python batch_aggregate.py --patients i4oK0F QZTsn6

# Search in different directory
python batch_aggregate.py --base-dir /path/to/data/
```

#### Example Output

```
Found 23 patient folder(s) to process:
  - i4oK0F
  - QZTsn6
  - sub-ABC123
  ...

Aggregation settings:
  Method: mean
  Include variability: False
  Distance metric: wasserstein
  Output suffix: aggregated

================================================================================
Processing 1/23: i4oK0F
================================================================================

Aggregating features for patient: i4oK0F
...

================================================================================
BATCH AGGREGATION COMPLETE
================================================================================

Results:
  Successful: 23/23
```

---

## Feature Types and Aggregation Strategies

### Scalar Features (Persistence Entropy, Summary Statistics)

**Method used:** Mean, Median, or Full statistics

**Example:**
- Input: 5 entropy values [2.1, 2.3, 2.2, 2.4, 2.0]
- Output (mean): `{'persistence_entropy_mean': 2.2, 'persistence_entropy_std': 0.14}`
- Output (full): `{'persistence_entropy_mean': 2.2, 'persistence_entropy_std': 0.14, 'persistence_entropy_min': 2.0, ...}`

### Persistence Diagrams

**Method used:** Medoid selection (Solution 5)

Selects the diagram that is most "central" to all others using Wasserstein distance.

**Example:**
- Input: 5 persistence diagrams
- Output: The diagram with index 2 (has minimum sum of distances to others)
- Also stores: medoid_index=2, mean_distance=0.15

### Array Features (Betti Curves, Persistence Landscapes, Heat Kernels)

**Method used:** Element-wise mean/median

**Example:**
- Input: 5 Betti curves, each of shape (1, 100, 4)
- Output: `{'betti_curve_mean': array of shape (1, 100, 4), 'betti_curve_std': array of shape (1, 100, 4)}`

### Distance Matrices

**No aggregation needed** - these are already computed across all slices.

---

## Variability Features (Optional)

When using `--include-variability`, additional features are computed that capture how much features vary across slices:

- **std**: Standard deviation
- **range**: Max - Min
- **iqr**: Interquartile range (Q75 - Q25)
- **cv**: Coefficient of variation (std / mean)

**Use case:** Test if medication affects signal stability/consistency.

**Example:**
```python
# Without variability
'H0_count_mean': 15.2

# With variability
'H0_count_mean': 15.2,
'H0_count_var_std': 2.1,
'H0_count_var_range': 8.0,
'H0_count_var_iqr': 3.5,
'H0_count_var_cv': 0.138
```

---

## Workflow

### Step 1: Extract Features (Already Done)

```bash
python feature_extraction.py data.mat events.txt ./i4oK0F/ --prefix medOff
python feature_extraction.py data.mat events.txt ./i4oK0F/ --prefix medOn
```

This creates:
- `medOff_left_hold_diagrams.pkl`
- `medOff_all_features.pkl`
- `medOn_all_features.pkl`
- ... etc

### Step 2: Aggregate Features

```bash
# Single patient
python aggregate_features.py ./i4oK0F/ --method mean

# All patients
python batch_aggregate.py --method mean
```

This creates:
- `i4oK0F/aggregated_features.pkl`
- `QZTsn6/aggregated_features.pkl`
- ... etc

### Step 3: Load Aggregated Features for Analysis

```python
import pickle
import pandas as pd

# Load aggregated features for one patient
with open('i4oK0F/aggregated_features.pkl', 'rb') as f:
    patient_data = pickle.load(f)

# Access specific features
medOn_left_hold = patient_data['medOn']['left_hold']
medOff_left_hold = patient_data['medOff']['left_hold']

print(f"MedOn entropy: {medOn_left_hold['persistence_entropy_mean']:.3f}")
print(f"MedOff entropy: {medOff_left_hold['persistence_entropy_mean']:.3f}")

# Create dataframe for statistical analysis
rows = []
for patient in all_patients:
    with open(f'{patient}/aggregated_features.pkl', 'rb') as f:
        data = pickle.load(f)

    for med_state in ['medOn', 'medOff']:
        for hemisphere in ['left', 'right']:
            for condition in ['hold', 'resting']:
                features = data[med_state][f'{hemisphere}_{condition}']

                row = {
                    'patient': patient,
                    'med_state': med_state,
                    'hemisphere': hemisphere,
                    'condition': condition,
                    **{k: v for k, v in features.items() if isinstance(v, (int, float))}
                }
                rows.append(row)

df = pd.DataFrame(rows)

# Now you can do statistical analysis
from scipy import stats

# Example: Compare persistence entropy between medOn and medOff
medOn_entropy = df[df['med_state']=='medOn']['persistence_entropy_mean']
medOff_entropy = df[df['med_state']=='medOff']['persistence_entropy_mean']

t_stat, p_value = stats.ttest_rel(medOn_entropy, medOff_entropy)
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
```

---

## Recommendations

### For Initial Exploration

```bash
python batch_aggregate.py --method mean
```

- Fast and simple
- Good for initial exploration
- Preserves mean and std for all features

### For Publication/Rigorous Analysis

```bash
python batch_aggregate.py --method full --include-variability
```

- Most comprehensive feature set
- Includes all statistics (mean, std, min, max, median, range, IQR)
- Includes variability metrics
- Allows testing multiple hypotheses

### For Robust Analysis (with Outliers)

```bash
python batch_aggregate.py --method median
```

- Robust to outliers
- Uses median and MAD instead of mean and std

---

## Troubleshooting

### Error: "No *_all_features.pkl files found"

**Cause:** Feature extraction hasn't been run yet.

**Solution:** Run `feature_extraction.py` first.

### Warning: "gtda not available"

**Cause:** giotto-tda not installed.

**Solution:** `pip install giotto-tda` or medoid selection will fall back to using first diagram.

### Error: "Error computing medoid"

**Cause:** Issue with persistence diagram format or distance computation.

**Solution:** The script will fall back to using the first diagram as representative.

---

## Technical Details

### Implemented Solutions from ANALYSIS_METHODOLOGY.md

- ✅ **Solution 1**: Averaging Scalar Features
- ✅ **Solution 2**: Statistical Summary (Mean + Std + Range)
- ✅ **Solution 3**: Median (Robust Central Tendency)
- ❌ **Solution 4**: Wasserstein Barycenter (skipped - requires additional libraries)
- ✅ **Solution 5**: Select Representative Slice (Medoid)
- ✅ **Solution 6**: Concatenate All Features (not implemented - use raw features if needed)
- ✅ **Solution 7**: PCA (not directly implemented - apply after aggregation if needed)
- ✅ **Solution 8**: Variability as Feature

### Performance

For typical datasets (5 slices per condition, ~100 points per diagram):

- Single patient: ~5-10 seconds
- All 23 patients: ~2-4 minutes

### Memory Usage

Peak memory usage: ~500MB per patient (depends on number and size of features)

---

## Next Steps

After aggregating features, proceed to:

1. **Exploratory Analysis** (ANALYSIS_METHODOLOGY.md Pathway 1-5)
2. **Statistical Testing** (Pathway 2, 7)
3. **Distance-Based Analysis** (Pathway 3)
4. **Visualization** (MDS plots, boxplots, etc.)

See `ANALYSIS_METHODOLOGY.md` for detailed analysis pathways.
