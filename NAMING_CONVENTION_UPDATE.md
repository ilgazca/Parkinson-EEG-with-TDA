# Naming Convention Update Summary

## Overview
Updated all scripts to follow a consistent naming convention that includes hold type indicators (holdL/holdR) and provides dominant/nondominant hemisphere mappings based on contralateral motor control.

## Key Changes

### 1. File Naming Convention
All feature files now include a hold type suffix:

**Pattern:**
- `{medState}_{hemisphere}_hold{L|R}.pkl` - Time series (e.g., `medOff_left_holdL.pkl`)
- `{medState}_{hemisphere}_hold{L|R}_diagrams.pkl` - Persistence diagrams
- `{medState}_all_features_hold{L|R}.pkl` - All extracted features
- `aggregated_features_hold{L|R}.pkl` - Aggregated features

**Examples:**
- `medOff_left_holdL.pkl` - Left channel, medOff state, left arm hold subject
- `medOn_right_holdR.pkl` - Right channel, medOn state, right arm hold subject
- `aggregated_features_holdL.pkl` - Aggregated features for left arm hold subject

### 2. Dominant/Non-Dominant Mapping
Based on **contralateral motor control** (crossed brain-body connections):

| Hold Type | Arm Raised | Dominant Hemisphere | Non-Dominant Hemisphere |
|-----------|------------|---------------------|-------------------------|
| holdL     | Left arm   | RIGHT hemisphere    | LEFT hemisphere         |
| holdR     | Right arm  | LEFT hemisphere     | RIGHT hemisphere        |

## Updated Scripts

### feature_extraction.py
- **Auto-detection**: Automatically detects HoldL or HoldR from .mat filename
- **Naming**: All output files include appropriate hold suffix
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

**Data Structure:**
```python
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

### Loading Aggregated Data
```python
import pickle

# Load aggregated features
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
2. **Simplified Analysis**: Use `dominant_hold` vs `nondominant_hold` for consistent cross-subject comparisons
3. **Contralateral Control**: Properly accounts for brain-body motor control relationships
4. **Backward Compatible**: Original left/right keys still available
5. **Automatic**: All detection and mapping happens automatically

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
