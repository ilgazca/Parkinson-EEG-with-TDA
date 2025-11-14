# Naming Convention Update Summary

## Overview
Updated all scripts to follow a consistent naming convention that includes hold type indicators (holdL/holdR) and provides dominant/nondominant hemisphere mappings based on contralateral motor control.

## Key Changes

### 1. File Naming Convention
All feature files now use dominant/nondominant hemisphere naming and include hold type suffix:

**Pattern (batch_feature_extraction.py - NEW):**
- `{medState}_{dominant|nondominant}_{condition}_{hold{L|R}}_diagrams.pkl` - Persistence diagrams
- `{medState}_{dominant|nondominant}_{condition}_{hold{L|R}}_diagram.png` - Diagram plots
- `{medState}_{dominant|nondominant}_{condition}_{hold{L|R}}_landscape.png` - Landscape plots
- `{medState}_{dominant|nondominant}_{condition}_{hold{L|R}}_betti.png` - Betti curve plots
- `{medState}_{dominant|nondominant}_{condition}_{hold{L|R}}_heat.png` - Heat kernel plots
- `{medState}_all_features_{hold{L|R}}.pkl` - All extracted features

**Examples (NEW convention):**
- `medOff_dominant_hold_holdL_diagrams.pkl` - Dominant hemisphere (right), medOff state, holdL subject
- `medOff_nondominant_hold_holdL_diagrams.pkl` - Non-dominant hemisphere (left), medOff state, holdL subject
- `medOn_dominant_resting_diagrams.pkl` - Dominant hemisphere, medOn state, resting period
- `medOn_nondominant_resting_diagrams.pkl` - Non-dominant hemisphere, medOn state, resting period
- `medOff_all_features_holdL.pkl` - All features for holdL subject, medOff state

**Pattern (legacy feature_extraction.py):**
- `{medState}_{hemisphere}_hold{L|R}.pkl` - Time series (e.g., `medOff_left_holdL.pkl`)
- `{medState}_{hemisphere}_hold{L|R}_diagrams.pkl` - Persistence diagrams
- `{medState}_all_features_hold{L|R}.pkl` - All extracted features
- `aggregated_features_hold{L|R}.pkl` - Aggregated features

### 2. Dominant/Non-Dominant Mapping
Based on **contralateral motor control** (crossed brain-body connections):

| Hold Type | Arm Raised | Dominant Hemisphere | Non-Dominant Hemisphere |
|-----------|------------|---------------------|-------------------------|
| holdL     | Left arm   | RIGHT hemisphere    | LEFT hemisphere         |
| holdR     | Right arm  | LEFT hemisphere     | RIGHT hemisphere        |

## Updated Scripts

### batch_feature_extraction.py (NEW)
- **Auto-detection**: Automatically detects HoldL or HoldR from .mat filename
- **Dominant mapping**: Maps left/right hemispheres to dominant/nondominant based on hold type
- **Naming**: All output files use dominant/nondominant naming for easier cross-subject comparison
- **Example output**:
  ```
  medOff_dominant_hold_holdL_diagrams.pkl
  medOff_nondominant_hold_holdL_diagrams.pkl
  medOff_dominant_resting_diagrams.pkl
  medOff_nondominant_resting_diagrams.pkl
  medOff_all_features_holdL.pkl
  ```
- **Benefits**:
  - Consistent naming across subjects regardless of which arm they raised
  - Dominant hemisphere always refers to the hemisphere controlling the raised arm
  - Simplifies analysis by allowing direct comparison of dominant vs nondominant features

### feature_extraction.py (LEGACY)
- **Auto-detection**: Automatically detects HoldL or HoldR from .mat filename
- **Naming**: All output files include appropriate hold suffix using left/right naming
- **Example output**:
  ```
  medOff_left_holdL.pkl
  medOff_left_holdL_diagrams.pkl
  medOff_all_features_holdL.pkl
  ```

### aggregate_features.py
- **Auto-detection**: Detects hold type from existing feature files
- **Dual naming**: Creates both left/right AND dominant/nondominant keys
- **Metadata**: Stores hold type information in `_metadata` key

**Data Structure (batch_feature_extraction.py output):**
```python
# Structure of medState_all_features_holdL.pkl
data = {
    'hold': {
        'dominant': {               # Active hemisphere controlling raised arm
            'persistence_entropy': ...,
            'stats': {...},
            'persistence_landscape': ...,
            'betti_curve': ...,
            'heat_kernel': ...
        },
        'nondominant': {            # Less-active hemisphere
            'persistence_entropy': ...,
            'stats': {...},
            'persistence_landscape': ...,
            'betti_curve': ...,
            'heat_kernel': ...
        }
    },
    'resting': {
        'dominant': {...},          # Same structure as above
        'nondominant': {...}
    }
}
```

**Data Structure (aggregate_features.py output):**
```python
# Structure of aggregated_features_holdL.pkl
data = {
    'medOn': {
        'left_hold': {...},           # Original left hemisphere, hold task
        'right_hold': {...},          # Original right hemisphere, hold task
        'left_resting': {...},
        'right_resting': {...},
        'dominant_hold': {...},       # Mapped to active hemisphere
        'nondominant_hold': {...},    # Mapped to less-active hemisphere
        'dominant_resting': {...},
        'nondominant_resting': {...}
    },
    'medOff': { ... },
    '_metadata': {
        'hold_type': 'holdL',
        'hold_suffix': '_holdL'
    }
}
```

### batch_aggregate.py
- **Updated glob pattern**: Now finds `*_all_features*.pkl` (includes hold suffixes)
- **Automatic**: Works seamlessly with updated aggregate_features.py

## Usage Examples

### Loading Feature Data (batch_feature_extraction.py output)
```python
import pickle

# Load features from batch_feature_extraction.py
with open('i4oK0F/medOff_all_features_holdL.pkl', 'rb') as f:
    data = pickle.load(f)

# Access features by condition and dominance
# For holdL subjects: dominant = right hemisphere, nondominant = left hemisphere
hold_dominant = data['hold']['dominant']  # Features from dominant (right) hemisphere during hold
hold_nondominant = data['hold']['nondominant']  # Features from nondominant (left) hemisphere during hold
resting_dominant = data['resting']['dominant']
resting_nondominant = data['resting']['nondominant']

# Access specific features
entropy = hold_dominant['persistence_entropy']
stats = hold_dominant['stats']
landscape = hold_dominant['persistence_landscape']
betti = hold_dominant['betti_curve']
heat = hold_dominant['heat_kernel']

print(f"Shape of persistence landscape: {landscape.shape}")
```

### Loading Aggregated Data (aggregate_features.py output - LEGACY)
```python
import pickle

# Load aggregated features (if using aggregate_features.py)
with open('i4oK0F/aggregated_features_holdL.pkl', 'rb') as f:
    data = pickle.load(f)

# Access by original hemisphere names
medOn_left_hold = data['medOn']['left_hold']
medOn_right_hold = data['medOn']['right_hold']

# Access by dominant/nondominant (recommended for analysis)
medOn_dominant_hold = data['medOn']['dominant_hold']  # RIGHT hemisphere for holdL
medOn_nondominant_hold = data['medOn']['nondominant_hold']  # LEFT hemisphere for holdL

# Check which hemisphere is dominant
print(f"Dominant hemisphere: {medOn_dominant_hold['hemisphere']}")  # 'right' for holdL
```

### Running Feature Extraction

**Using batch_feature_extraction.py (RECOMMENDED):**
```bash
# Process all 14 patients automatically
python batch_feature_extraction.py

# Outputs for each patient (e.g., i4oK0F with holdL):
# - medOff_dominant_hold_holdL_diagrams.pkl
# - medOff_nondominant_hold_holdL_diagrams.pkl
# - medOff_dominant_resting_diagrams.pkl
# - medOff_nondominant_resting_diagrams.pkl
# - medOff_all_features_holdL.pkl
# (and corresponding .png plot files)
```

**Using feature_extraction.py (LEGACY):**
```bash
# The script auto-detects hold type from filename
python feature_extraction.py \
    sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/sub-i4oK0F_HoldL_MedOff.mat \
    i4oK0F/event_times.txt \
    ./i4oK0F/ \
    --prefix medOff

# Outputs: medOff_all_features_holdL.pkl, medOff_left_holdL_diagrams.pkl, etc.
```

### Running Aggregation
```bash
# Single patient
python aggregate_features.py ./i4oK0F/ --method mean

# Batch processing
python batch_aggregate.py --method mean
```

## Benefits

1. **Clear Subject Identification**: Immediately know which arm each subject raised
2. **Simplified Analysis**: Use `dominant` vs `nondominant` keys for consistent cross-subject comparisons
3. **Contralateral Control**: Properly accounts for brain-body motor control relationships
   - holdL subjects: dominant = right hemisphere (controls left arm)
   - holdR subjects: dominant = left hemisphere (controls right arm)
4. **Consistent Naming**: All subjects can be compared using the same key names regardless of which arm they raised
5. **Automatic Detection**: All hold type detection and dominant/nondominant mapping happens automatically
6. **Research-Oriented**: File names directly reflect the neurological significance (dominant hemisphere = active during task)

## Migration Notes

### Existing Data
- Existing .pkl files with old naming (e.g., `medOff_all_features.pkl`) are still supported
- The cleanup script has renamed existing files to follow new convention
- Files already processed: 0cGdk9, i4oK0F, jyC0j3, QZTsn6

### New Data
- All new feature extractions will automatically use the new naming convention
- Hold type is auto-detected from .mat filenames
- No manual intervention required

## Subject Hold Type Reference

From the data directory:
- **holdL subjects**: 0cGdk9, 2IU8mi, AB2PeX, AbzsOg, FYbcap, PuPVlx, QZTsn6, dCsWjQ, gNX5yb, i4oK0F
- **holdR subjects**: 2IhVOz, BYJoWR, VopvKx, jyC0j3
