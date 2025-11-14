# Analysis Pipeline for MedOn vs MedOff Comparison

## Overview

This document outlines the analysis strategy for comparing topological features between medication states (medOn vs medOff) in Parkinson's disease patients using TDA methods. The goal is to identify discriminative features **without using machine learning models** initially.

## Data Structure

### Available Data
- **14 patients** with extracted topological features
- **2 medication states**: medOn, medOff (9 patients with both, 3 with only medOff, 2 with only medOn)
- **2 hemispheres**: left (non-dominant or dominant), right (dominant or non-dominant)
- **2 conditions**: resting, hold task
- **4 homology dimensions**: H0, H1, H2, H3

### Feature Files
For each patient with available data:
- `{medState}_all_features_{holdType}.pkl` - Contains all extracted features
- `{medState}_{hemisphere}_{condition}_{holdType}_diagrams.pkl` - Raw persistence diagrams

## Feature Types and Comparison Strategies

### 1. Scalar Features (Summary Statistics)

#### Features Included
- **Feature counts** per homology dimension (H0, H1, H2, H3)
- **Lifespan statistics**: mean, max, std per dimension
- **Birth time statistics**: mean per dimension
- **Death time statistics**: mean per dimension
- **Persistence entropy**: single scalar value per diagram

#### Comparison Strategy

**Step 1: Organize Data**
```python
# Structure: patient → medState → hemisphere → condition → features
scalar_features = {
    'H0_count': [], 'H1_count': [], 'H2_count': [], 'H3_count': [],
    'H0_mean_lifespan': [], 'H0_max_lifespan': [], 'H0_std_lifespan': [],
    # ... repeat for H1, H2, H3
    'persistence_entropy': [],
    'patient_id': [], 'med_state': [], 'hemisphere': [], 'condition': []
}
```

**Step 2: Statistical Tests**
For each scalar feature:

1. **Paired t-test** (for patients with both medOn and medOff):
   - Compare medOn vs medOff for same patient/hemisphere/condition
   - Use `scipy.stats.ttest_rel()`
   - Reports: t-statistic, p-value, effect size (Cohen's d)

2. **Wilcoxon signed-rank test** (non-parametric alternative):
   - Use when data is not normally distributed
   - Use `scipy.stats.wilcoxon()`

3. **Effect size calculation**:
   - Cohen's d: `(mean_medOn - mean_medOff) / pooled_std`
   - Interpret: small (0.2), medium (0.5), large (0.8)

**Step 3: Visualization**
- **Paired scatter plots**: Each point is a patient, x=medOn, y=medOff
- **Box plots**: Side-by-side comparison of distributions
- **Violin plots**: Show full distribution shape
- **Raincloud plots**: Combine box plot, violin, and individual points

**Scaling Considerations**:
- **Generally NOT needed** for statistical tests (paired comparisons preserve relationships)
- **May be useful** for visualization if features have very different ranges
- If scaling, use **StandardScaler** after organizing data by feature type

---

### 2. Persistence Landscapes

#### Features Included
- Multi-dimensional arrays representing persistent features
- Separate landscapes for each homology dimension (H0, H1, H2, H3)
- Typically 100 samples per layer × N layers

#### Comparison Strategy

**Step 1: Aggregate Landscapes**
Since landscapes are functional representations:
```python
# Option A: Compare landscape layers individually
# Extract first few layers (e.g., first 3 layers) for each dimension

# Option B: Compute landscape norms
L1_norm = np.sum(np.abs(landscape))
L2_norm = np.sqrt(np.sum(landscape**2))
Linf_norm = np.max(np.abs(landscape))
```

**Step 2: Distance-Based Comparison**
1. **L2 distance between landscapes**:
   - `dist = np.linalg.norm(landscape_medOn - landscape_medOff)`
   - Compute per patient/hemisphere/condition/dimension
   - Test if distances from medOn→medOff differ from expected

2. **Functional ANOVA**:
   - Treat landscapes as functional data
   - Compare curve shapes between conditions

**Step 3: Pointwise Comparison**
- Compute mean landscape across patients for each condition
- Identify regions where medOn and medOff landscapes diverge
- Use **pointwise t-tests** with multiple comparison correction (Bonferroni or FDR)

**Step 4: Visualization**
- **Mean ± SEM curves**: Plot average landscape with error bands
- **Heatmaps**: Show landscape matrices for medOn vs medOff
- **Difference plots**: medOn - medOff landscape

**Scaling Considerations**:
- **Critical**: Landscapes should already be on comparable scales
- If comparing across dimensions, may need **normalization by L2 norm**
- Consider **standardizing within each dimension** before cross-dimension comparisons

---

### 3. Betti Curves

#### Features Included
- Curves showing the number of topological features at each filtration value
- One curve per homology dimension (H0, H1, H2, H3)
- Typically 100 filtration values

#### Comparison Strategy

**Step 1: Curve Similarity Metrics**
1. **Area under curve (AUC)**:
   - `auc = np.trapz(betti_curve, filtration_values)`
   - Compare AUC between medOn and medOff

2. **Maximum Betti number**:
   - `max_betti = np.max(betti_curve)`
   - When does maximum occur? `argmax_filtration = filtration[np.argmax(betti_curve)]`

3. **Curve integrals and moments**:
   - First moment: centroid of Betti curve
   - Second moment: spread/variance

**Step 2: Statistical Comparison**
- Use AUC, max Betti, and moments as scalar features
- Apply paired t-tests as in Section 1
- Compare distributions across patients

**Step 3: Functional Data Analysis**
- Similar to persistence landscapes
- Pointwise comparison at each filtration value
- Identify filtration ranges where curves differ most

**Step 4: Visualization**
- **Overlaid curves**: medOn vs medOff on same plot
- **Mean curves with confidence bands**
- **Difference curves**: medOn - medOff Betti numbers

**Scaling Considerations**:
- **Usually NOT needed**: Betti numbers are counts (already on same scale)
- Exception: If using curve derivatives or higher-order features

---

### 4. Heat Kernel Signatures

#### Features Included
- Time-evolved representation of persistence diagrams
- Matrix form capturing diffusion process
- Sensitive to both birth/death times and persistence

#### Comparison Strategy

**Step 1: Vectorize Heat Kernels**
- Flatten heat kernel matrices into vectors
- Or extract summary statistics (trace, eigenvalues, etc.)

**Step 2: Distance Metrics**
1. **Frobenius norm**: `np.linalg.norm(kernel_medOn - kernel_medOff, 'fro')`
2. **Spectral distance**: Compare eigenvalue spectra

**Step 3: Statistical Tests**
- Convert to scalar summaries (trace, dominant eigenvalue, etc.)
- Apply paired tests as in Section 1

**Step 4: Visualization**
- **Heatmap comparison**: Side-by-side heat kernel matrices
- **Eigenvalue spectra**: Plot eigenvalues for medOn vs medOff
- **Difference matrices**: medOn - medOff heat kernels

**Scaling Considerations**:
- **May be needed** if kernel values span very different ranges
- Use **MinMaxScaler** to normalize kernel matrices to [0,1]
- Or **normalize by total heat** (sum of all kernel values)

---

## Multi-Level Analysis Strategy

### Level 1: Within-Patient Comparisons (Primary Analysis)

**Focus**: How do features change within the same patient between medOn and medOff?

**Advantages**:
- Controls for inter-patient variability
- Most powerful for detecting medication effects
- Uses paired statistical tests

**Analysis Steps**:
1. For each patient with both medOn and medOff data:
   - Compute difference: Δ = feature_medOn - feature_medOff
   - Analyze across hemisphere and condition
2. Test if Δ is significantly different from zero (one-sample t-test)
3. Report: mean Δ, 95% CI, p-value, effect size

**Considerations**:
- Only 9 patients have both medication states
- Still analyze separately by hemisphere (left/right) and condition (resting/hold)

---

### Level 2: Group-Level Comparisons (Secondary Analysis)

**Focus**: Are there consistent patterns across all patients?

**Advantages**:
- Can include all 14 patients (if analyzing medOn and medOff separately)
- Identifies population-level trends

**Analysis Steps**:
1. Aggregate features across all patients for each condition:
   - Group 1: All medOn samples
   - Group 2: All medOff samples
2. Independent t-tests or Mann-Whitney U tests
3. Visualize distributions

**Considerations**:
- Higher inter-patient variability may obscure effects
- Use as supporting evidence, not primary conclusion

---

### Level 3: Hemisphere-Specific Analysis

**Focus**: Are medication effects lateralized?

**Analysis Steps**:
1. Compare dominant vs non-dominant hemisphere:
   - Remember: holdL subjects → right hemisphere dominant
   - holdR subjects → left hemisphere dominant
2. Test interaction: Does medication effect differ by hemisphere?
3. Use **repeated-measures ANOVA** or **mixed-effects models**:
   - Within-subject factors: medState, hemisphere, condition
   - Random effect: patient

**Visualization**:
- Interaction plots showing medOn vs medOff for each hemisphere
- Separate analysis by dominant/non-dominant

---

### Level 4: Condition-Specific Analysis (Resting vs Hold)

**Focus**: Do medication effects differ during rest vs active tasks?

**Analysis Steps**:
1. Compare resting vs hold separately for medOn and medOff
2. Test interaction: Is medOn-medOff difference larger during hold or rest?
3. Use 2×2 within-subject ANOVA:
   - Factor 1: medState (medOn, medOff)
   - Factor 2: condition (resting, hold)

**Research Questions**:
- Does medication have stronger effects during motor activity?
- Are topological changes task-dependent?

---

## Recommended Analysis Workflow

### Phase 1: Exploratory Data Analysis (EDA)

**Goal**: Understand data structure and identify interesting patterns

1. **Load all feature files**:
   ```python
   import pickle
   import pandas as pd

   # Create master dataframe
   all_features = []
   for patient_id in patient_list:
       for med_state in ['medOn', 'medOff']:
           # Load features and organize
   ```

2. **Compute summary statistics**:
   - Mean, median, std, range for each feature
   - Check for outliers or missing data
   - Visualize distributions (histograms)

3. **Correlation analysis**:
   - Which features are correlated within medOn or medOff?
   - Do correlations change between medication states?

4. **Visualize with PCA**:
   - Project high-dimensional features to 2D
   - Color by medState, separate plots by hemisphere/condition
   - **Scale features before PCA**: StandardScaler

---

### Phase 2: Feature-Specific Statistical Tests

**Goal**: Identify which specific features differ between medOn and medOff

1. **For each scalar feature**:
   - Paired t-test (within-patient comparison)
   - Compute effect size
   - Create paired scatter plots
   - Report: p-value, effect size, confidence intervals

2. **For each array-based feature** (landscapes, Betti curves):
   - Compute summary metrics (AUC, norms, etc.)
   - Apply statistical tests to summaries
   - Perform pointwise comparisons
   - Visualize average curves with error bands

3. **Multiple comparison correction**:
   - Use **Bonferroni correction** for conservative approach
   - Or **FDR (Benjamini-Hochberg)** for less conservative
   - Adjusted α = 0.05 / number_of_tests

---

### Phase 3: Homology Dimension Analysis

**Goal**: Determine which homology dimensions are most discriminative

1. **Compare effect sizes across dimensions**:
   - H0 (connected components): Basic connectivity
   - H1 (loops/cycles): Oscillatory patterns
   - H2 (voids/cavities): Higher-order structure
   - H3 (3D voids): Complex multi-dimensional patterns

2. **Rank dimensions by discriminative power**:
   - Which dimension shows largest effect size?
   - Which has smallest p-value?
   - Which is most consistent across patients?

3. **Interpret biological meaning**:
   - H0: Overall signal fragmentation
   - H1: Periodic dynamics (oscillations)
   - H2/H3: Complex multi-variate patterns

---

### Phase 4: Multi-Factor Analysis

**Goal**: Understand interactions between factors (hemisphere, condition, dimension)

1. **Repeated-measures ANOVA**:
   ```python
   # Factors: medState, hemisphere, condition, dimension
   # Test main effects and interactions
   ```

2. **Mixed-effects models**:
   ```python
   import statsmodels.api as sm
   from statsmodels.formula.api import mixedlm

   # Fixed effects: medState, hemisphere, condition
   # Random effect: patient (intercept)
   model = mixedlm("feature ~ medState * hemisphere * condition",
                   data, groups=data["patient_id"])
   ```

3. **Post-hoc tests**:
   - If interaction is significant, perform pairwise comparisons
   - Use Tukey HSD or Bonferroni correction

---

### Phase 5: Integrative Summary

**Goal**: Synthesize findings across all analyses

1. **Create summary table**:
   - Rows: Features
   - Columns: Effect size, p-value, direction of change
   - Highlight significant findings

2. **Rank features by importance**:
   - Criteria: effect size, consistency, biological interpretability

3. **Generate comprehensive visualizations**:
   - Multi-panel figures showing top discriminative features
   - Forest plots showing effect sizes with confidence intervals

---

## Practical Implementation Guide

### Step-by-Step Code Structure

```python
# 1. LOAD DATA
def load_all_features(patient_ids, base_path):
    """
    Load all feature files and organize into structured format
    Returns: pandas DataFrame with columns:
    - patient_id, med_state, hemisphere, condition,
    - all scalar features, array-based feature references
    """
    pass

# 2. COMPUTE SUMMARY METRICS FOR ARRAY FEATURES
def compute_landscape_summaries(landscapes):
    """Extract AUC, norms, peak values from landscapes"""
    pass

def compute_betti_summaries(betti_curves):
    """Extract AUC, max Betti, centroid from curves"""
    pass

# 3. STATISTICAL TESTS
def paired_comparison(df, feature_name):
    """
    Perform paired t-test for patients with both medOn/medOff
    Returns: t_stat, p_value, cohen_d, confidence_interval
    """
    pass

def multiple_test_correction(p_values, method='fdr_bh'):
    """Apply FDR or Bonferroni correction"""
    from statsmodels.stats.multitest import multipletests
    return multipletests(p_values, method=method)

# 4. VISUALIZATION
def plot_paired_scatter(df, feature_name):
    """Scatter plot with medOn vs medOff, one point per patient"""
    pass

def plot_feature_distributions(df, feature_name):
    """Box/violin plot comparing medOn vs medOff"""
    pass

def plot_landscape_comparison(landscapes_medOn, landscapes_medOff):
    """Mean curves with confidence bands"""
    pass

# 5. COMPREHENSIVE ANALYSIS
def run_full_analysis(df):
    """
    Execute all statistical tests and generate summary report
    """
    results = {}
    for feature in scalar_features:
        results[feature] = paired_comparison(df, feature)

    # Apply multiple comparison correction
    p_values = [r['p_value'] for r in results.values()]
    corrected = multiple_test_correction(p_values)

    # Create summary table
    summary = create_summary_table(results, corrected)
    return summary
```

---

## Scaling Guidelines by Feature Type

### When to Scale

| Feature Type | Scale Before Statistical Tests? | Scale Before PCA/Visualization? | Recommended Scaler |
|--------------|-------------------------------|--------------------------------|-------------------|
| **Feature counts** (H0-H3) | ❌ No | ✅ Yes | StandardScaler |
| **Lifespan statistics** | ❌ No | ✅ Yes | StandardScaler |
| **Persistence entropy** | ❌ No | ✅ Yes | StandardScaler |
| **Persistence landscapes** | ⚠️ Maybe* | ✅ Yes | StandardScaler or L2 norm |
| **Betti curves** | ❌ No | ⚠️ Maybe** | None or StandardScaler |
| **Heat kernels** | ⚠️ Maybe* | ✅ Yes | MinMaxScaler or L2 norm |

*If comparing across homology dimensions (H0 vs H1 vs H2 vs H3)
**Only if comparing across different conditions/patients

### Scaling Implementation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For scalar features (before PCA only)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# For landscapes (normalize by L2 norm per diagram)
landscape_norm = landscape / np.linalg.norm(landscape)

# For heat kernels (normalize to [0,1])
kernel_scaled = (kernel - kernel.min()) / (kernel.max() - kernel.min())
```

---

## Expected Outcomes

### Positive Findings (What to Look For)

1. **Significant p-values** (< 0.05 after correction) for paired tests
2. **Medium to large effect sizes** (Cohen's d > 0.5)
3. **Consistent directionality** across patients (most show same trend)
4. **Specific homology dimensions** standing out (e.g., H1 most discriminative)
5. **Lateralization patterns** (e.g., dominant hemisphere shows stronger effects)
6. **Task-specific effects** (e.g., larger differences during hold than rest)

### Null Findings (What to Do)

1. **No significant differences in primary tests**:
   - Check for ceiling/floor effects
   - Try non-parametric tests (Wilcoxon)
   - Examine individual patient trajectories

2. **High variability obscuring effects**:
   - Stratify by clinical covariates (disease duration, severity)
   - Use mixed-effects models to partition variance

3. **Inconsistent effects across patients**:
   - Identify responders vs non-responders
   - Look for subgroups with distinct patterns

---

## Reporting Guidelines

### Minimal Reporting Standards

For each significant finding, report:

1. **Descriptive statistics**:
   - Mean ± SD for medOn and medOff
   - Median and IQR if non-normal

2. **Test statistics**:
   - Test used (paired t-test, Wilcoxon, etc.)
   - Test statistic value
   - Degrees of freedom
   - p-value (raw and corrected)

3. **Effect size**:
   - Cohen's d with 95% CI
   - Interpretation (small/medium/large)

4. **Sample size**:
   - Number of patients analyzed
   - Number with both medOn and medOff

5. **Visualization**:
   - At least one plot per significant finding
   - Show individual patient data when possible

---

## Advanced Considerations

### Confounding Variables to Consider

1. **Disease severity**: UPDRS scores (if available)
2. **Disease duration**: Years since diagnosis
3. **Age and sex**: May affect baseline topology
4. **Time of day**: When recording was taken
5. **Medication dosage**: Amount of L-DOPA or other medications

**If available**, include as covariates in mixed-effects models.

---

### Power Analysis

With only 9 patients having both medOn and medOff:

- **Detectable effect size** (α=0.05, power=0.80): Cohen's d ≈ 0.95 (large)
- **Consider**:
  - Focus on features with large, consistent effects
  - Use less conservative corrections (FDR instead of Bonferroni)
  - Report effect sizes even for non-significant findings
  - This is exploratory analysis → discovery mode

---

## Summary Checklist

Before concluding analysis:

- [ ] All 14 patients' data loaded and organized
- [ ] Summary statistics computed for all features
- [ ] Paired statistical tests performed for scalar features
- [ ] Array-based features converted to summary metrics
- [ ] Multiple comparison correction applied
- [ ] Effect sizes calculated and interpreted
- [ ] Homology dimensions ranked by discriminative power
- [ ] Hemisphere and condition effects examined
- [ ] Visualizations created for top findings
- [ ] Results table summarizing all tests
- [ ] Biological interpretation drafted

---

## References and Further Reading

### Statistical Methods
- **Paired t-tests**: Suitable for within-subject designs
- **Effect sizes**: Cohen, J. (1988). Statistical Power Analysis
- **Multiple comparisons**: Benjamini & Hochberg (1995). FDR method
- **Mixed-effects models**: Bates et al. (2015). lme4 package

### TDA-Specific Analysis
- **Persistence landscapes**: Bubenik, P. (2015). Statistical topological data analysis
- **Wasserstein distances**: Turner et al. (2014). Persistent homology transform
- **Functional data analysis**: Ramsay & Silverman (2005). Functional Data Analysis

### Parkinson's Disease
- Consider consulting domain literature on:
  - Expected effects of dopaminergic medication on neural oscillations
  - Lateralization of motor symptoms
  - LFP/EEG signatures in PD

---

## Conclusion

This pipeline provides a structured approach to comparing topological features between medication states. Start with **paired within-patient comparisons** (most powerful), then expand to **group-level** and **multi-factor analyses**. Focus on **effect sizes** and **consistency** as much as p-values, given the moderate sample size. Visualize liberally and interpret findings in the context of Parkinson's disease neurobiology.

**Key Principle**: Let the data guide you, but maintain rigor in statistical testing and multiple comparison correction.
