# Analysis Methodology: Distinguishing MedOn vs MedOff States Using TDA

## Project Goal
Manually distinguish between medication-on (medOn) and medication-off (medOff) states in Parkinson's disease patients using topological data analysis features, **without machine learning**, through direct comparison and statistical analysis.

## Dataset Structure
- **23 patients** with LFP/MEG recordings
- **2 medication states**: medOn, medOff
- **2 brain hemispheres**: left, right
- **2 conditions**: resting state, hold task
- **Total combinations per patient**: 2 (med states) × 2 (hemispheres) × 2 (conditions) = 8 recordings

## Available Topological Features
From `feature_extraction.py`, we extract:
1. **Persistence Diagrams** (H0, H1, H2, H3)
2. **Persistence Entropy**
3. **Summary Statistics** (feature count, mean/std/min/max lifespan, birth/death times)
4. **Persistence Landscapes**
5. **Betti Curves**
6. **Heat Kernel Signatures**
7. **Distance Matrices** (Wasserstein, Bottleneck)

---

## PREPROCESSING: Combining Multiple Slices into Single Representative Features

### The Problem
The `feature_extraction.py` script processes multiple time slices per condition (e.g., 5 hold events and 5 resting events per hemisphere), resulting in multiple persistence diagrams and feature vectors for each state-condition combination. Before conducting medOn vs medOff comparisons, we need to aggregate these multiple slices into a single representative feature vector/matrix per patient, hemisphere, condition, and medication state.

**Example scenario:**
- Patient `sub-i4oK0F`, left hemisphere, hold condition, medOff state
- Input: 5 persistence diagrams (one per hold event slice)
- Goal: Obtain 1 representative persistence diagram or feature vector
- This process must be repeated for all combinations: 2 hemispheres × 2 conditions × 2 med states = 8 aggregations per patient

---

### Solution 1: Averaging Scalar Features

**Best for:** Persistence entropy, summary statistics, scalar-valued features

#### Approach
Take the mean (or median) of each scalar feature across all slices.

#### Pros
- Simple and interpretable
- Reduces noise by averaging
- Works well for normally distributed features
- Preserves physical meaning of features

#### Cons
- Loses information about variability across slices
- Assumes slices are independent and identically distributed
- Sensitive to outliers (if using mean)

#### Implementation
```python
import numpy as np
import pickle

# Load features for all slices of a condition
with open("patient_X/medOff_left_hold_diagrams.pkl", "rb") as f:
    diagrams = pickle.load(f)  # List of diagrams, one per slice

# Extract summary statistics for each slice
from eeg_utils import extract_features

slice_features = []
for diagram in diagrams:
    features = extract_features(diagram, homology_dimensions=[0, 1, 2, 3])
    slice_features.append(features)

# Average across slices
aggregated_features = {}
for key in slice_features[0].keys():
    values = [sf[key] for sf in slice_features]
    aggregated_features[key + '_mean'] = np.mean(values)
    aggregated_features[key + '_std'] = np.std(values)  # Also keep variability info

print(aggregated_features)
# Example output: {'H0_count_mean': 15.2, 'H0_count_std': 2.1, ...}
```

#### Recommended for
- Persistence entropy values
- Feature counts (H0_count, H1_count, etc.)
- Mean/max lifespans
- Mean birth/death times

---

### Solution 2: Statistical Summary (Mean + Std + Range)

**Best for:** When variability information is important

#### Approach
Extract multiple statistics (mean, std, min, max, median) across slices for each feature.

#### Pros
- Preserves information about variability
- More comprehensive representation
- Can capture both central tendency and spread
- Useful if some patients have high variability while others don't

#### Cons
- Increases feature dimensionality
- May introduce redundancy

#### Implementation
```python
def aggregate_with_statistics(slice_features):
    """
    Aggregate features across slices with multiple statistics.

    Args:
        slice_features: List of feature dictionaries (one per slice)

    Returns:
        Dictionary with aggregated features (mean, std, min, max, median)
    """
    aggregated = {}

    for key in slice_features[0].keys():
        values = np.array([sf[key] for sf in slice_features])

        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
        aggregated[f"{key}_min"] = np.min(values)
        aggregated[f"{key}_max"] = np.max(values)
        aggregated[f"{key}_median"] = np.median(values)
        aggregated[f"{key}_range"] = np.max(values) - np.min(values)

    return aggregated

# Usage
aggregated = aggregate_with_statistics(slice_features)
```

#### Recommended for
- Exploratory analysis to see if variability differs between medOn/medOff
- When you want to test hypotheses about stability (e.g., "medication reduces variability")

---

### Solution 3: Median (Robust Central Tendency)

**Best for:** When slices may have outliers

#### Approach
Use median instead of mean for robust estimation.

#### Pros
- Robust to outliers
- Better when slice quality varies
- Still interpretable

#### Cons
- Loses some information compared to mean
- Less efficient statistically if data is normally distributed

#### Implementation
```python
# Simply replace np.mean with np.median in Solution 1
aggregated_features[key + '_median'] = np.median(values)
```

---

### Solution 4: Wasserstein Barycenter for Persistence Diagrams

**Best for:** Persistence diagrams (the most principled approach)

#### Approach
Compute the Wasserstein barycenter (geometric mean) of multiple persistence diagrams. This is the diagram that minimizes the sum of squared Wasserstein distances to all input diagrams.

#### Pros
- Theoretically sound for averaging persistence diagrams
- Preserves topological structure
- Results in a single representative persistence diagram
- Respects the geometry of diagram space

#### Cons
- Computationally expensive
- Requires specialized libraries
- May not have closed-form solution
- Complexity increases with number of diagrams

#### Implementation
```python
# Note: Wasserstein barycenter requires specialized libraries
# giotto-tda doesn't have built-in barycenter, but we can approximate

from scipy.optimize import minimize
from gtda.diagrams import PairwiseDistance, Scaler

def approximate_diagram_barycenter(diagrams, metric='wasserstein'):
    """
    Approximate the barycenter of multiple persistence diagrams.

    This is a simplified approach: find the diagram that has minimum
    sum of distances to all other diagrams (medoid).
    """
    # Extract 2D diagrams
    diagrams_2d = [d[0] for d in diagrams]

    # Pad to same size
    from eeg_utils import pad_diagrams
    diagrams_padded = pad_diagrams(diagrams_2d)

    # Scale
    scaler = Scaler(metric=metric)
    diagrams_scaled = scaler.fit_transform(diagrams_padded)

    # Compute pairwise distances
    pwise = PairwiseDistance(metric=metric)
    dist_matrix = pwise.fit_transform(diagrams_scaled)[0]

    # Find medoid (diagram with minimum sum of distances)
    sum_distances = np.sum(dist_matrix, axis=1)
    medoid_idx = np.argmin(sum_distances)

    return diagrams[medoid_idx], medoid_idx

# Usage
barycenter_diagram, idx = approximate_diagram_barycenter(diagrams)
print(f"Representative diagram is slice {idx}")
```

#### Advanced Implementation
For true Wasserstein barycenter, consider using external libraries:
- **POT (Python Optimal Transport)**: Has `ot.bregman.barycenter` function
- **persim**: Persistence diagram utilities

```python
# Using POT library (requires: pip install POT)
import ot

def wasserstein_barycenter_diagrams(diagrams, weights=None):
    """
    Compute Wasserstein barycenter using POT library.
    Note: This is a conceptual example; actual implementation
    requires careful preprocessing of diagrams.
    """
    # Convert diagrams to point clouds
    # ... preprocessing ...

    # Compute barycenter
    if weights is None:
        weights = np.ones(len(diagrams)) / len(diagrams)

    barycenter = ot.lp.free_support_barycenter(diagrams, weights)
    return barycenter
```

#### Recommended for
- When you want to preserve the full persistence diagram structure
- Distance-based analysis (Pathway 3)
- Visualization of "average" diagram

---

### Solution 5: Select Representative Slice (Medoid)

**Best for:** When you want to keep an actual observed diagram

#### Approach
Select the slice that is most "central" or "representative" based on distances to all other slices.

#### Pros
- Results in actual observed data (not synthetic average)
- Computationally simpler than barycenter
- Preserves real topological structure

#### Cons
- Throws away information from other slices
- May be sensitive to which distance metric is used
- Doesn't utilize all available data

#### Implementation
```python
def select_representative_slice(diagrams, metric='wasserstein'):
    """
    Select the diagram that is most representative (medoid).

    Returns:
        Representative diagram and its index
    """
    from gtda.diagrams import PairwiseDistance, Scaler
    from eeg_utils import pad_diagrams

    # Pad and scale
    diagrams_2d = [d[0] for d in diagrams]
    diagrams_padded = pad_diagrams(diagrams_2d)
    scaler = Scaler(metric=metric)
    diagrams_scaled = scaler.fit_transform(diagrams_padded)

    # Compute pairwise distances
    pwise = PairwiseDistance(metric=metric)
    dist_matrix = pwise.fit_transform(diagrams_scaled)[0]

    # Find medoid
    sum_distances = np.sum(dist_matrix, axis=1)
    medoid_idx = np.argmin(sum_distances)

    return diagrams[medoid_idx], medoid_idx, sum_distances

# Usage
representative, idx, distances = select_representative_slice(diagrams)
print(f"Selected slice {idx} as representative")
print(f"Average distance to other slices: {distances[idx]:.4f}")
```

---

### Solution 6: Concatenate All Features (Keep All Information)

**Best for:** When you want to preserve all slice information

#### Approach
Stack features from all slices into a single long vector.

#### Pros
- No information loss
- Preserves temporal/sequential information if slices are ordered
- Can capture patterns across slices

#### Cons
- High dimensionality (features × n_slices)
- Requires equal number of slices across all recordings (or padding)
- May include noise
- Harder to interpret

#### Implementation
```python
def concatenate_slice_features(slice_features):
    """
    Concatenate all slice features into a single vector.

    Args:
        slice_features: List of feature dictionaries

    Returns:
        Single concatenated feature dictionary
    """
    concatenated = {}

    for slice_idx, features in enumerate(slice_features):
        for key, value in features.items():
            concatenated[f"{key}_slice{slice_idx}"] = value

    return concatenated

# Usage
concatenated = concatenate_slice_features(slice_features)
# Result: {'H0_count_slice0': 15, 'H0_count_slice1': 18, ...}
```

#### Alternative: Flatten arrays
```python
# For array features like Betti curves, persistence landscapes
import numpy as np

all_betti_curves = []  # Shape: (n_slices, n_points, n_dimensions)
for diagram in diagrams:
    bc = BettiCurve().fit_transform(diagram)
    all_betti_curves.append(bc[0])  # Remove batch dimension

# Flatten
flattened = np.concatenate([bc.flatten() for bc in all_betti_curves])
```

---

### Solution 7: PCA or Dimensionality Reduction

**Best for:** High-dimensional features like landscapes, Betti curves

#### Approach
Apply PCA to reduce dimensionality while preserving variance.

#### Pros
- Reduces noise
- Captures main patterns
- Lower dimensionality aids visualization
- Works well for high-dimensional features

#### Cons
- Loses interpretability
- Requires fitting PCA model
- May miss nonlinear patterns

#### Implementation
```python
from sklearn.decomposition import PCA

def reduce_dimensionality_across_slices(features_list, n_components=5):
    """
    Apply PCA to reduce feature dimensionality across slices.

    Args:
        features_list: List of feature arrays (one per slice)
        n_components: Number of principal components

    Returns:
        Reduced feature vector
    """
    # Stack features
    X = np.vstack(features_list)  # Shape: (n_slices, n_features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    # Average across slices or take first PC
    aggregated = np.mean(X_reduced, axis=0)

    return aggregated, pca.explained_variance_ratio_

# Usage for Betti curves
all_betti = [BC.fit_transform(d)[0].flatten() for d in diagrams]
reduced, variance = reduce_dimensionality_across_slices(all_betti, n_components=5)
print(f"Variance explained: {variance}")
```

---

### Solution 8: Variability as Feature

**Best for:** Testing if medication affects signal stability

#### Approach
Instead of aggregating to central tendency, use the variability itself as a feature.

#### Pros
- Captures stability/instability differences
- May reveal that medication affects consistency rather than mean values
- Provides complementary information

#### Cons
- Requires sufficient number of slices
- Interpretation is different from typical features

#### Implementation
```python
def extract_variability_features(slice_features):
    """
    Extract variability measures across slices as features.

    Returns:
        Dictionary of variability metrics
    """
    variability = {}

    for key in slice_features[0].keys():
        values = np.array([sf[key] for sf in slice_features])

        # Standard deviation
        variability[f"{key}_std"] = np.std(values)

        # Coefficient of variation (normalized variability)
        mean_val = np.mean(values)
        if mean_val != 0:
            variability[f"{key}_cv"] = np.std(values) / mean_val

        # Range
        variability[f"{key}_range"] = np.max(values) - np.min(values)

        # Interquartile range (robust measure)
        variability[f"{key}_iqr"] = np.percentile(values, 75) - np.percentile(values, 25)

    return variability

# Usage
var_features = extract_variability_features(slice_features)

# Test hypothesis: Does medication reduce variability?
medOn_var = extract_variability_features(medOn_slice_features)
medOff_var = extract_variability_features(medOff_slice_features)
```

---

### Solution 9: Hierarchical Aggregation

**Best for:** When you want multi-level representation

#### Approach
Create features at multiple levels: individual slices, pairs of slices, all slices.

#### Pros
- Multi-scale representation
- Captures both local (single slice) and global (all slices) patterns
- Flexible

#### Cons
- Complex
- High dimensionality
- May be overkill for this application

---

### Solution 10: Domain-Specific Aggregation

**Best for:** When you have additional information about slices

#### Approach
Weight slices differently based on quality metrics, signal-to-noise ratio, or temporal position.

#### Implementation
```python
def weighted_average_features(slice_features, weights):
    """
    Compute weighted average of features across slices.

    Args:
        slice_features: List of feature dictionaries
        weights: Array of weights (must sum to 1)

    Returns:
        Weighted aggregated features
    """
    aggregated = {}

    for key in slice_features[0].keys():
        values = np.array([sf[key] for sf in slice_features])
        aggregated[key] = np.sum(values * weights)

    return aggregated

# Example: Weight by signal quality or distance from artifact
weights = np.array([0.15, 0.25, 0.3, 0.2, 0.1])  # Higher weight for middle slices
aggregated = weighted_average_features(slice_features, weights)
```

---

## Recommended Aggregation Strategy

### For Different Feature Types

| Feature Type | Primary Method | Alternative Method | Rationale |
|-------------|----------------|-------------------|-----------|
| **Persistence Diagrams** | Medoid (Solution 5) | Barycenter (Solution 4) | Medoid is simpler; barycenter if computational resources allow |
| **Persistence Entropy** | Mean (Solution 1) | Mean + Std (Solution 2) | Scalar values average well |
| **Summary Statistics** | Mean (Solution 1) | Median (Solution 3) if outliers present | Simple and interpretable |
| **Betti Curves** | Mean curve + Std bands (Solution 2) | PCA (Solution 7) | Preserves curve shape |
| **Persistence Landscapes** | Mean (Solution 1) | PCA (Solution 7) for high-dim | Landscapes are designed to be averaged |
| **Heat Kernel** | Mean (Solution 1) | - | Similar to landscapes |

### Practical Implementation Pipeline

```python
import numpy as np
import pickle
from pathlib import Path

def aggregate_all_features(patient_folder, condition, hemisphere, med_state):
    """
    Complete aggregation pipeline for one patient-condition-hemisphere-med_state combination.

    Args:
        patient_folder: Path to patient directory
        condition: 'hold' or 'resting'
        hemisphere: 'left' or 'right'
        med_state: 'medOn' or 'medOff'

    Returns:
        Dictionary of aggregated features
    """
    prefix = f"{med_state}_{hemisphere}_{condition}"

    # Load persistence diagrams
    diagrams_path = Path(patient_folder) / f"{prefix}_diagrams.pkl"
    with open(diagrams_path, 'rb') as f:
        diagrams = pickle.load(f)

    # Load all features
    features_path = Path(patient_folder) / f"{med_state}_all_features.pkl"
    with open(features_path, 'rb') as f:
        all_features = pickle.load(f)

    aggregated = {}

    # 1. Persistence diagrams: Use medoid
    representative_diagram, rep_idx = select_representative_slice(
        diagrams, metric='wasserstein'
    )
    aggregated['persistence_diagram'] = representative_diagram
    aggregated['representative_slice_idx'] = rep_idx

    # 2. Persistence entropy: Mean and Std
    pe_values = all_features[f'{hemisphere}_{condition}_pe']
    aggregated['persistence_entropy_mean'] = np.mean(pe_values)
    aggregated['persistence_entropy_std'] = np.std(pe_values)

    # 3. Summary statistics: Mean
    stats_list = all_features[f'{hemisphere}_{condition}_stats']
    for key in stats_list[0].keys():
        values = [s[key] for s in stats_list]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)

    # 4. Betti curves: Mean curve
    bc_values = all_features[f'{hemisphere}_{condition}_bc']
    aggregated['betti_curve_mean'] = np.mean(bc_values, axis=0)
    aggregated['betti_curve_std'] = np.std(bc_values, axis=0)

    # 5. Persistence landscapes: Mean
    pl_values = all_features[f'{hemisphere}_{condition}_pl']
    aggregated['persistence_landscape_mean'] = np.mean(pl_values, axis=0)

    # 6. Heat kernel: Mean
    hk_values = all_features[f'{hemisphere}_{condition}_hk']
    aggregated['heat_kernel_mean'] = np.mean(hk_values, axis=0)

    return aggregated

# Usage: Create aggregated dataset for all patients
all_patients = ['i4oK0F', 'QZTsn6', ...]  # List of patient IDs
aggregated_dataset = []

for patient in all_patients:
    for med_state in ['medOn', 'medOff']:
        for hemisphere in ['left', 'right']:
            for condition in ['hold', 'resting']:
                agg = aggregate_all_features(
                    f"./{patient}/",
                    condition,
                    hemisphere,
                    med_state
                )
                agg.update({
                    'patient': patient,
                    'med_state': med_state,
                    'hemisphere': hemisphere,
                    'condition': condition
                })
                aggregated_dataset.append(agg)

# Save aggregated dataset
with open("aggregated_dataset.pkl", "wb") as f:
    pickle.dump(aggregated_dataset, f)
```

---

### Decision Guide: Which Method to Use?

```
START
  │
  ├─ Is feature a persistence diagram?
  │   YES → Use Medoid (Solution 5) or Barycenter (Solution 4)
  │   NO → Continue
  │
  ├─ Is feature scalar (single number)?
  │   YES → Use Mean (Solution 1)
  │   NO → Continue
  │
  ├─ Is feature high-dimensional (>50 features)?
  │   YES → Consider PCA (Solution 7)
  │   NO → Continue
  │
  ├─ Is feature a curve/function (Betti, Landscape)?
  │   YES → Use Mean curve (Solution 1 or 2)
  │   NO → Use Mean (Solution 1)
  │
  └─ Always consider keeping variability info (Std, Range)
```

---

## Pathway 1: Visual Comparison of Persistence Diagrams

### Approach
Compare persistence diagrams directly through visualization to identify qualitative differences.

### Steps
1. **Single Patient Comparison**
   - Plot medOn vs medOff persistence diagrams side-by-side for same hemisphere and condition
   - Separate plots for H0, H1, H2, H3
   - Look for differences in:
     - Number of features
     - Distribution of points
     - Lifespan ranges
     - Clustering patterns

2. **Aggregate Visualization**
   - Overlay all medOn diagrams (semi-transparent) vs all medOff diagrams
   - Create density plots to show concentration areas
   - Separate by condition (resting vs hold) and hemisphere

3. **Key Questions**
   - Are there more/fewer topological features in one medication state?
   - Do features have longer/shorter lifespans in medOn vs medOff?
   - Are there specific regions in birth-death space that differentiate the states?

### Implementation
```python
# Visualize all persistence diagrams for comparison
import matplotlib.pyplot as plt
from gtda.plotting import plot_diagram

# Load diagrams for one patient
medOn_diagrams = pickle.load(open("patient_X/medOn_left_hold_diagrams.pkl", "rb"))
medOff_diagrams = pickle.load(open("patient_X/medOff_left_hold_diagrams.pkl", "rb"))

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_diagram(medOn_diagrams[0], axes=axes[0])
axes[0].set_title("MedOn - Left Hold")
plot_diagram(medOff_diagrams[0], axes=axes[1])
axes[1].set_title("MedOff - Left Hold")
```

---

## Pathway 2: Statistical Comparison of Summary Features

### Approach
Extract numerical summaries from persistence diagrams and compare using statistical tests.

### Steps
1. **Feature Extraction**
   - For each recording, extract per homology dimension:
     - Number of features
     - Mean lifespan
     - Max lifespan
     - Mean birth time
     - Mean death time
     - Std of lifespans

2. **Statistical Testing**
   - **Paired t-test** (within-subject): Compare medOn vs medOff for each patient
   - **Mann-Whitney U test** (non-parametric): If data not normally distributed
   - **Effect size calculation**: Cohen's d to quantify magnitude of difference

3. **Aggregation Levels**
   - **Per patient**: Does each patient show consistent direction of change?
   - **Per hemisphere**: Are effects lateralized?
   - **Per condition**: Are effects stronger in resting or hold?
   - **Per homology dimension**: Which dimensions (H0, H1, H2, H3) show strongest effects?

### Implementation
```python
import numpy as np
from scipy import stats
import pandas as pd

# Create dataframe of features
data = []
for patient in all_patients:
    for med_state in ['medOn', 'medOff']:
        features = extract_features(diagrams, homology_dimensions=[0,1,2,3])
        data.append({
            'patient': patient,
            'med_state': med_state,
            'hemisphere': 'left',
            'condition': 'hold',
            'h0_count': features['H0_count'],
            'h0_mean_lifespan': features['H0_mean_lifespan'],
            'h1_count': features['H1_count'],
            # ... etc
        })

df = pd.DataFrame(data)

# Paired t-test for each feature
for feature in ['h0_count', 'h0_mean_lifespan', 'h1_count', ...]:
    medOn_values = df[df['med_state']=='medOn'][feature]
    medOff_values = df[df['med_state']=='medOff'][feature]
    t_stat, p_value = stats.ttest_rel(medOn_values, medOff_values)
    print(f"{feature}: t={t_stat:.3f}, p={p_value:.4f}")
```

---

## Pathway 3: Distance Matrix Analysis

### Approach
Use topological distances (Wasserstein, Bottleneck) to assess whether medOn and medOff recordings cluster separately.

### Steps
1. **Compute Pairwise Distances**
   - Calculate distances between all recordings across all patients
   - Use both Wasserstein and Bottleneck metrics
   - Separate by homology dimension

2. **Intra-group vs Inter-group Comparison**
   - **Intra-medOn**: Average distance between all medOn recordings
   - **Intra-medOff**: Average distance between all medOff recordings
   - **Inter-group**: Average distance between medOn and medOff recordings
   - **Hypothesis**: If states are distinguishable, inter-group distances should be larger

3. **Visualization**
   - Multidimensional scaling (MDS) to visualize distance matrix in 2D
   - Color code by medication state
   - Look for clustering/separation

4. **Per-Patient Analysis**
   - For each patient, compute distance between their medOn and medOff recordings
   - Compare to average intra-group distances
   - Is within-subject medOn-medOff distance larger than typical within-state variability?

### Implementation
```python
from gtda.diagrams import PairwiseDistance, Scaler
from sklearn.manifold import MDS

# Combine all diagrams
all_diagrams = medOn_diagrams + medOff_diagrams
labels = ['medOn']*len(medOn_diagrams) + ['medOff']*len(medOff_diagrams)

# Pad and scale
diagrams_padded = pad_diagrams([d[0] for d in all_diagrams])
scaler = Scaler(metric='wasserstein')
diagrams_scaled = scaler.fit_transform(diagrams_padded)

# Compute distances
pwise = PairwiseDistance(metric='wasserstein')
distance_matrix = pwise.fit_transform(diagrams_scaled)

# MDS visualization
mds = MDS(n_components=2, dissimilarity='precomputed')
embedding = mds.fit_transform(distance_matrix[0])

plt.scatter(embedding[:, 0], embedding[:, 1], c=[0 if l=='medOn' else 1 for l in labels])
plt.legend(['medOn', 'medOff'])
```

---

## Pathway 4: Persistence Entropy Analysis

### Approach
Compare the "complexity" or "randomness" of persistence diagrams using entropy.

### Steps
1. **Extract Entropy Values**
   - Compute persistence entropy for each recording
   - Separate by homology dimension (H0, H1, H2, H3)

2. **Distribution Comparison**
   - Plot distributions (histograms, boxplots) of entropy for medOn vs medOff
   - Visual assessment of separation

3. **Statistical Testing**
   - Paired t-test or Wilcoxon signed-rank test
   - Test hypothesis: medOn entropy ≠ medOff entropy

4. **Per-Dimension Analysis**
   - Which homology dimensions show largest entropy differences?
   - Interpretation: What does higher/lower entropy mean for brain dynamics?

### Implementation
```python
from gtda.diagrams import PersistenceEntropy

PE = PersistenceEntropy()

# Extract entropies
medOn_entropies = [PE.fit_transform(d) for d in medOn_diagrams]
medOff_entropies = [PE.fit_transform(d) for d in medOff_diagrams]

# Plot distributions
plt.figure(figsize=(8, 5))
plt.boxplot([medOn_entropies, medOff_entropies], labels=['medOn', 'medOff'])
plt.ylabel('Persistence Entropy')
plt.title('Entropy Comparison')

# Statistical test
t_stat, p_value = stats.ttest_rel(medOn_entropies, medOff_entropies)
print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
```

---

## Pathway 5: Betti Curves Analysis

### Approach
Compare Betti curves to assess differences in topological complexity over filtration.

### Steps
1. **Extract Betti Curves**
   - Betti curves show number of topological features at each filtration value
   - Separate curves for H0, H1, H2, H3

2. **Visual Comparison**
   - Plot average Betti curves for medOn vs medOff
   - Include confidence bands (mean ± std)
   - Identify regions of filtration where curves diverge

3. **Quantitative Comparison**
   - Compute area under curve (AUC) for each Betti curve
   - Compare AUC between medication states
   - Compute maximum Betti number differences

4. **Curve Distance Metrics**
   - L1 or L2 distance between Betti curves
   - Test if medOn curves are more similar to each other than to medOff curves

### Implementation
```python
from gtda.diagrams import BettiCurve

BC = BettiCurve()

# Compute Betti curves
medOn_betti = [BC.fit_transform(d) for d in medOn_diagrams]
medOff_betti = [BC.fit_transform(d) for d in medOff_diagrams]

# Plot average curves
medOn_avg = np.mean(medOn_betti, axis=0)
medOff_avg = np.mean(medOff_betti, axis=0)

plt.plot(medOn_avg[0, :, 1], label='medOn H1')  # H1 example
plt.plot(medOff_avg[0, :, 1], label='medOff H1')
plt.xlabel('Filtration value')
plt.ylabel('Betti number')
plt.legend()
```

---

## Pathway 6: Persistence Landscape Comparison

### Approach
Use persistence landscapes as functional representations and compare their shapes.

### Steps
1. **Extract Landscapes**
   - Compute persistence landscapes for all recordings
   - Landscapes are continuous functions that encode diagram information

2. **Functional Analysis**
   - Compute L^p norms of landscapes (p=1, 2, ∞)
   - Compare norm distributions between medOn and medOff

3. **Landscape Visualization**
   - Plot average landscapes for each medication state
   - Identify regions where landscapes differ most

4. **Statistical Testing**
   - Compare landscape norms using t-tests
   - Use landscape features as discriminative measures

---

## Pathway 7: Within-Subject (Paired) Analysis

### Approach
Use each patient as their own control to reduce inter-subject variability.

### Steps
1. **Paired Differences**
   - For each patient, compute difference between medOn and medOff features
   - This eliminates baseline individual differences

2. **Consistency Analysis**
   - Count how many patients show increase vs decrease in each feature
   - Sign test: Is the direction of change consistent across patients?

3. **Magnitude Analysis**
   - Average absolute difference across patients
   - Identify features with largest consistent changes

4. **Visualization**
   - Slopegraphs: Connect medOn and medOff values for each patient
   - Highlight consistent vs inconsistent patterns

### Implementation
```python
# For each patient, compute delta
deltas = []
for patient in all_patients:
    medOn_feature = get_feature(patient, 'medOn', 'h1_count')
    medOff_feature = get_feature(patient, 'medOff', 'h1_count')
    delta = medOn_feature - medOff_feature
    deltas.append(delta)

# Count direction
n_positive = np.sum(np.array(deltas) > 0)
n_negative = np.sum(np.array(deltas) < 0)
print(f"Patients with increase: {n_positive}, decrease: {n_negative}")

# Sign test
from scipy.stats import binom_test
p_value = binom_test(n_positive, len(deltas), 0.5)
print(f"Sign test p-value: {p_value:.4f}")
```

---

## Pathway 8: Stratified Analysis by Hemisphere and Condition

### Approach
Analyze effects separately for different recording contexts.

### Steps
1. **Hemisphere-Specific Analysis**
   - Separate analysis for left and right hemisphere
   - Question: Are medication effects lateralized?
   - Compare effect sizes between hemispheres

2. **Condition-Specific Analysis**
   - Separate analysis for resting vs hold
   - Question: Are effects stronger during active task or rest?
   - Compare effect sizes between conditions

3. **Interaction Analysis**
   - Does hemisphere affect how medication impacts topology?
   - Does condition affect medication response?
   - 2×2×2 comparison (med state × hemisphere × condition)

---

## Pathway 9: Multi-Feature Discriminant Score

### Approach
Combine multiple topological features into a single discriminant score (without ML).

### Steps
1. **Feature Standardization**
   - Z-score normalize all features
   - Ensures equal weighting

2. **Simple Linear Combination**
   - Average of standardized features that show significant differences
   - Or: weighted average based on effect sizes

3. **Discriminant Score per Recording**
   - Compute score for each medOn and medOff recording
   - Compare score distributions

4. **Threshold-Based Classification**
   - Find optimal threshold that separates medOn from medOff
   - Report accuracy, sensitivity, specificity
   - This is manual classification without ML training

---

## Pathway 10: Time-Series of Topological Features

### Approach
If you have multiple slices per condition, analyze temporal patterns.

### Steps
1. **Temporal Trajectories**
   - Plot feature values over time/slice number
   - Compare trajectories between medOn and medOff

2. **Variability Analysis**
   - Compare temporal variability (std over slices) between states
   - Question: Is one state more stable than the other?

3. **Correlation Analysis**
   - Correlation between features across time
   - Do different features co-vary differently under medication?

---

## Recommended Analysis Pipeline

### Phase 1: Exploratory (Visual)
1. Start with **Pathway 1** (persistence diagrams visualization)
2. Look at **Pathway 4** (entropy distributions)
3. Examine **Pathway 5** (Betti curves)

### Phase 2: Statistical Testing
4. Conduct **Pathway 2** (summary statistics comparison)
5. Perform **Pathway 7** (within-subject paired analysis)

### Phase 3: Distance-Based Analysis
6. Execute **Pathway 3** (distance matrix and clustering)
7. Use MDS or PCA for visualization

### Phase 4: Stratification
8. Repeat key analyses with **Pathway 8** (by hemisphere and condition)

### Phase 5: Integration
9. Combine findings into **Pathway 9** (discriminant score)
10. Evaluate overall separability

---

## Key Considerations

### Statistical Power
- With 23 patients, you have reasonable power for detecting medium-to-large effects
- Use paired tests where possible to increase power
- Consider false discovery rate (FDR) correction if testing many features

### Multiple Comparisons
- You'll be testing many features (H0, H1, H2, H3 × multiple statistics)
- Apply Bonferroni or FDR correction to control Type I error

### Effect Size Reporting
- Always report effect sizes (Cohen's d, r, etc.) alongside p-values
- Small effects may be statistically significant but not practically meaningful

### Reproducibility
- Document all parameters and preprocessing steps
- Save intermediate results
- Create reproducible scripts for each analysis pathway

### Interpretation
- Connect topological findings back to neuroscience
  - H0: Connected components → network fragmentation?
  - H1: Loops → oscillatory/cyclic dynamics?
  - H2, H3: Higher-order structures → complex coordination?

---

## Expected Outputs

### Visualization Outputs
1. Persistence diagram comparison plots
2. Betti curve overlays
3. MDS/PCA scatter plots with medication state coloring
4. Boxplots/violin plots of summary statistics
5. Heatmaps of distance matrices

### Statistical Outputs
1. Table of t-test/Mann-Whitney results for each feature
2. Effect sizes with confidence intervals
3. Sign test results for within-subject consistency
4. Correlation matrices

### Summary Outputs
1. Classification accuracy using simple threshold
2. Feature importance ranking (by effect size)
3. Per-patient summary of medication effects
4. Recommendations for most discriminative features/dimensions

---

## Next Steps

1. **Data Collection**: Run `feature_extraction.py` on all 23 patients
2. **Create Analysis Scripts**: Implement pathways in Jupyter notebook or Python scripts
3. **Systematic Testing**: Work through pathways in recommended order
4. **Documentation**: Keep detailed notes on findings
5. **Refinement**: Based on initial results, focus on most promising pathways
