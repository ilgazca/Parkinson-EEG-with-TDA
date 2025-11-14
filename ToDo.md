# Analysis Infrastructure To-Do List

This checklist tracks the implementation of analysis infrastructure to support the statistical comparison of medOn vs medOff conditions as outlined in `Analysis_Pipeline.md`.

## Phase 1: Core Utilities (Foundation)

### Data Loading
- [x] Create `data_loader.py` module
  - [x] Function to load all patients' feature files
  - [x] Function to organize features into pandas DataFrame
  - [x] Function to handle missing data (patients with only medOn or medOff)
  - [x] Function to extract specific feature types (scalars, landscapes, etc.)
  - [x] Validation checks for data integrity

### Statistical Analysis
- [x] Create `analysis_utils.py` module
  - [x] Paired t-test function with effect size (Cohen's d)
  - [x] Wilcoxon signed-rank test (non-parametric alternative)
  - [x] Multiple comparison correction (FDR and Bonferroni)
  - [x] Independent t-test for group-level comparisons
  - [x] Summary table generation function
  - [x] Array feature summarization functions:
    - [x] Persistence landscape summaries (AUC, norms, peaks)
    - [x] Betti curve summaries (AUC, max Betti, centroid)
    - [x] Heat kernel summaries (Frobenius norm, trace)

### Visualization
- [x] Create `visualization_utils.py` module
  - [x] Paired scatter plot (medOn vs medOff per patient)
  - [x] Box plot and violin plot for distributions
  - [x] Forest plot for effect sizes with confidence intervals
  - [x] Raincloud plot (combined distribution visualization)
  - [x] Landscape comparison plot (mean curves with error bands)
  - [x] Betti curve comparison plot
  - [x] Heatmap for heat kernel comparisons
  - [x] Multi-panel summary figure generator

---

## Phase 2: Directory Structure

### Results Organization
- [x] Create `results/` directory structure
  - [x] `results/exploratory/` - For EDA outputs
  - [x] `results/statistical_tests/` - For test results and summary tables
  - [x] `results/figures/` - For publication-quality figures
    - [x] `results/figures/scalar_features/`
    - [x] `results/figures/landscapes/`
    - [x] `results/figures/betti_curves/`
    - [x] `results/figures/heat_kernels/`
    - [x] `results/figures/summary/`
  - [x] `results/reports/` - For analysis reports and summaries

---

## Phase 3: Configuration and Documentation

### Configuration
- [ ] Create `analysis_config.yaml`
  - [ ] Patient lists (all 14 patients, paired only, medOn only, medOff only)
  - [ ] Feature names (scalar features, array features)
  - [ ] Statistical parameters (alpha level, correction method)
  - [ ] Plotting parameters (figure size, colors, style)
  - [ ] Hemisphere mapping (dominant/nondominant by patient)

### Documentation
- [ ] Create `ANALYSIS_README.md`
  - [ ] Quick-start guide for analysis workflow
  - [ ] How to use data_loader
  - [ ] How to use analysis_utils
  - [ ] How to use visualization_utils
  - [ ] Example code snippets
  - [ ] Troubleshooting guide

- [x] Update `CLAUDE.md`
  - [x] Document new analysis structure
  - [x] Add sections on analysis utilities
  - [x] Update workflow to include analysis phase

---

## Phase 4: Analysis Notebooks

### Structured Analysis Workflow
- [x] Create `01_Exploratory_Analysis.ipynb`
  - [x] Load all patient data
  - [x] Compute descriptive statistics
  - [x] Check distributions (histograms, Q-Q plots)
  - [x] Normality tests (Shapiro-Wilk) for pooled and hemisphere-specific data
  - [x] Hemisphere-specific distribution analysis (dominant vs nondominant)
  - [x] Quantitative lateralization comparison
  - [ ] Identify outliers (statistical outlier detection)
  - [ ] Correlation analysis within conditions
  - [ ] PCA visualization

- [ ] Create `02_Statistical_Tests.ipynb`
  - [ ] Paired t-tests for all scalar features
  - [ ] Effect size calculations
  - [ ] Multiple comparison correction
  - [ ] Summary table of significant findings
  - [ ] Paired scatter plots for top features

- [ ] Create `03_Homology_Dimension_Analysis.ipynb`
  - [ ] Compare H0, H1, H2, H3 discriminative power
  - [ ] Effect sizes by dimension
  - [ ] Visualization of dimension-specific patterns
  - [ ] Interpretation of biological meaning

- [ ] Create `04_Multi_Factor_Analysis.ipynb`
  - [ ] Hemisphere effects (dominant vs nondominant)
  - [ ] Condition effects (resting vs hold)
  - [ ] Repeated-measures ANOVA
  - [ ] Mixed-effects models
  - [ ] Interaction plots

- [ ] Create `05_Final_Summary.ipynb`
  - [ ] Comprehensive results visualization
  - [ ] Top discriminative features ranked
  - [ ] Forest plots for all significant findings
  - [ ] Integrated summary figures
  - [ ] Export results for publication

---

## Phase 5: Optional Automation (Future)

### Batch Processing Scripts
- [ ] Create `run_paired_tests.py`
  - [ ] Automated statistical testing on all features
  - [ ] Save results to CSV/Excel
  - [ ] Generate summary plots automatically

- [ ] Create `generate_summary_report.py`
  - [ ] Compile all analysis results
  - [ ] Generate HTML or PDF report
  - [ ] Include tables and figures

---

## Testing and Validation

- [ ] Test data_loader on all 14 patients
- [ ] Verify statistical functions with known test cases
- [ ] Check visualization functions produce correct plots
- [ ] Validate results against manual calculations
- [ ] Test with subset of patients before full analysis

---

## Current Priority

**Phase 4: Analysis Notebooks** (IN PROGRESS)

✅ Completed:
- Phase 1: Core Utilities (data_loader, analysis_utils, visualization_utils)
- Phase 2: Results directory structure

🔄 Current Focus:
- `01_Exploratory_Analysis.ipynb`: Distribution analysis complete, working on outlier detection and correlation analysis
  - Completed: Data loading, descriptive statistics, normality checks, hemisphere-specific analyses
  - Remaining: Outlier detection, correlation analysis, PCA visualization

📋 Next Steps:
1. Complete remaining sections of `01_Exploratory_Analysis.ipynb`
2. Begin `02_Statistical_Tests.ipynb` for hypothesis testing
3. Progress through remaining analysis notebooks (03, 04, 05)

---

## Notes

- Take things slowly and test each component before moving forward
- Test on a single patient or small subset before running full analysis
- Document assumptions and decisions as you go
- Save intermediate results frequently
- Keep code modular and reusable

---

**Last Updated**: 2025-11-14 (Updated with Phase 4 progress)
