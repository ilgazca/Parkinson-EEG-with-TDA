
import pandas as pd
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from typing import Tuple, Optional, List
from gtda.time_series import SingleTakensEmbedding
from gtda.time_series import TakensEmbedding

def downsample_eeg_dataframe(df: pd.DataFrame, original_fs: float, target_fs: float, plot: bool = True) -> pd.DataFrame:
    """
    Downsamples an EEG DataFrame to a target sampling frequency.

    Args:
        df (pd.DataFrame): The input DataFrame where each column is an EEG channel
                          and the index represents time.
        original_fs (float): The original sampling frequency of the EEG data.
        target_fs (float): The desired target sampling frequency.
        plot (bool, optional): If True, plots the downsampled data. Defaults to True.

    Returns:
        pd.DataFrame: The downsampled DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(original_fs, (int, float)) or original_fs <= 0:
        raise ValueError("Input 'original_fs' must be a positive number.")
    if not isinstance(target_fs, (int, float)) or target_fs <= 0:
        raise ValueError("Input 'target_fs' must be a positive number.")
    if target_fs > original_fs:
        raise ValueError("Target sampling frequency cannot be greater than the original sampling frequency.")

    downsample_factor = int(original_fs / target_fs)

    if original_fs % target_fs != 0:
        print(f"Warning: Target Fs ({target_fs}) is not an even divisor of {original_fs}.")
        print(f"Using factor {downsample_factor}, which gives ~{original_fs / downsample_factor} Hz")

    print(f"Downsampling by a factor of {downsample_factor}...")

    # Apply the 'decimate' function to each column (channel)
    # decimate requires integer downsampling factor
    df_downsampled = df.apply(lambda x: decimate(x, downsample_factor), axis=0)

    # Re-create the time index for the new DataFrame
    new_num_samples = len(df_downsampled)
    new_fs = original_fs / downsample_factor

    new_time_vector = np.linspace(
        0,
        (new_num_samples - 1) / new_fs,
        new_num_samples
    )
    df_downsampled.index = pd.Index(new_time_vector, name="Time (s)")

    print("--- Downsampled DataFrame ---")
    print(df_downsampled.head())

    print("--- New DataFrame Info ---")
    df_downsampled.info()

    if plot:
        # Plot the new, downsampled data
        # Limiting to first 500 samples for better visualization, adjust as needed
        plot_data = df_downsampled.iloc[:min(500, len(df_downsampled))]
        plot_data.plot(
            subplots=True,
            layout=(len(plot_data.columns), 1) if len(plot_data.columns) > 0 else (1,1), # Adjust layout dynamically
            grid=True,
            title=f"Downsampled LFP/EEG Time Series ({new_fs:.2f} Hz)",
            figsize=(15, 2 * len(plot_data.columns)) # Adjust figure size dynamically
        )
        plt.tight_layout()
        plt.show()

    return df_downsampled


def mat_to_dataframe(file_path: str) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Loads a .mat file containing LFP data, converts it to a pandas DataFrame
    with a time index, and extracts left and right LFP channels.

    This function automatically detects keys for data, labels, and sampling rate.

    Args:
        file_path (str): The full path to the .mat file.

    Returns:
        A tuple containing:
        - pd.DataFrame: The main DataFrame with all signals.
        - pd.Series: The left LFP signal.
        - pd.Series: The right LFP signal.
        Returns None if the file cannot be processed or LFP channels are not found.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at '{file_path}'")
        return None

    print(f"--- Processing file: {os.path.basename(file_path)} ---")
    mat = loadmat(file_path)

    # --- Automatically find the data, labels, and Fs keys ---
    data_key, labels_key, fs_key = None, None, None

    # 1. Find the data matrix key by looking for a 2D array with many samples
    for key, value in mat.items():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] < value.shape[1] and value.shape[
            0] < 20:
            data_key = key
            print(f"Found data matrix. Key: '{data_key}', Shape: {value.shape}")
            break

    # 2. Find the labels key
    # Prioritize 'target_labels', but search for others if not found.
    possible_label_keys = ['target_labels', 'labels', 'channel_labels', 'ch_names']
    for key in possible_label_keys:
        if key in mat:
            labels_key = key
            print(f"Found labels. Key: '{labels_key}'")
            break

    # 3. Find the sampling rate key
    possible_fs_keys = ['fs', 'Fs', 'fsample', 'sampling_rate', 'srate']
    for key in possible_fs_keys:
        if key in mat:
            fs_key = key
            print(f"Found sampling rate. Key: '{fs_key}'")
            break

    # --- Error checking ---
    if data_key is None:
        print("\nERROR: Could not automatically find the data matrix.")
        return None
    if labels_key is None:
        print("\nERROR: Could not find labels key.")
        return None

    # --- Extract data, labels, and Fs ---
    # Transpose from (channels, time) to (time, channels)
    signals = mat[data_key].T

    # This pattern handles MATLAB cell arrays of strings
    try:
        labels = [str(label[0]) for label in mat[labels_key][0]]
    except (TypeError, IndexError):
        # Fallback for simple string arrays or other formats
        labels = [str(label).strip() for label in mat[labels_key].flatten()]
    print(f"Labels extracted: {labels}")

    fs = float(mat[fs_key][0, 0]) if fs_key else 2000.0  # Default to 2000.0 if not found
    if not fs_key:
        print("\nWARNING: Could not find sampling rate (Fs). Defaulting to 2000.0 Hz.")

    # --- Create DataFrame ---
    num_samples = signals.shape[0]
    time_vector = np.linspace(0, (num_samples - 1) / fs, num_samples)
    index = pd.Index(time_vector, name="Time (s)")

    df = pd.DataFrame(signals, columns=labels, index=index)
    print("\n--- DataFrame Created Successfully ---")
    print(df.head())

    # --- Automatically find and extract LFP channels ---
    left_lfp_col = next((col for col in df.columns if 'LFP-left' in col), None)
    right_lfp_col = next((col for col in df.columns if 'LFP-right' in col), None)

    if not left_lfp_col or not right_lfp_col:
        print("\nERROR: Could not find one or both LFP channels in the DataFrame columns.")
        return None

    left_lfp = df[left_lfp_col]
    right_lfp = df[right_lfp_col]

    print(f"\nSuccessfully extracted left LFP channel: '{left_lfp_col}'")
    print(f"Successfully extracted right LFP channel: '{right_lfp_col}'")

    return df, left_lfp, right_lfp


def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """
    Fits a Takens embedder and displays optimal search parameters.
    """
    try:
        y_embedded = embedder.fit_transform(y)
    except ValueError as e:
        if "Expected n_neighbors <= n_samples" in str(e):
            raise ValueError(
                "The input time series is likely too short for the Takens embedding "
                "parameter search. Please provide a longer time series or "
                "adjust the 'dimension' and 'time_delay' parameters of the "
                "SingleTakensEmbedding."
            ) from e
        else:
            raise

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )

    return y_embedded

def filter_persistence_diagram(diagram_3d: np.ndarray, lifespan_threshold: float) -> np.ndarray:
    """
    Filters a 3D persistence diagram to remove topological features with a lifespan
    below a given threshold.

    Args:
        diagram_3d (np.ndarray): The input persistence diagram with shape (n_samples, n_points, 3).
                                 Typically the output of VietorisRipsPersistence.fit_transform().
        lifespan_threshold (float): The minimum lifespan a feature must have to be kept.

    Returns:
        np.ndarray: A filtered 3D persistence diagram with the same shape format as the input.
    """
    if not isinstance(diagram_3d, np.ndarray) or diagram_3d.ndim != 3:
        raise TypeError("Input 'diagram_3d' must be a 3D NumPy array.")

    # Assumes n_samples is 1, which is common for single time series analysis
    diagram_2d = diagram_3d[0]

    # Calculate lifespans (death - birth)
    lifespans = diagram_2d[:, 1] - diagram_2d[:, 0]

    # Create a boolean mask
    mask = lifespans > lifespan_threshold

    # Apply the mask to the 2D diagram
    filtered_diagram_2d = diagram_2d[mask]

    print(f"Original number of topological features: {len(diagram_2d)}")
    print(f"Number of features after filtering: {len(filtered_diagram_2d)}")

    # Reshape back to 3D for compatibility with plotting functions
    return filtered_diagram_2d[None, :, :]


def extract_features(diagram: np.ndarray, homology_dimensions=[0, 1, 2], verbose: bool = True) -> dict:
    """
    Extracts summary statistics from a persistence diagram for ML.

    Args:
        diagram (np.ndarray): A persistence diagram from giotto-tda.
                              Can be a 3D array of shape (1, n_features, 3)
                              or a 2D array of shape (n_features, 3) where
                              columns are [birth, death, dimension].
        homology_dimensions (tuple): The dimensions to compute features for.
        verbose (bool): If True, prints the extracted features in a formatted way.

    Returns:
        dict: A flat dictionary of features, suitable for ML.
    """
    # --- Handle both 2D and 3D diagram inputs ---
    if diagram.ndim == 3:
        # Extract the 2D diagram from the 3D container (assuming n_samples=1)
        diagram_2d = diagram[0]
    elif diagram.ndim == 2:
        diagram_2d = diagram
    else:
        raise ValueError(f"Input diagram must be 2D or 3D, but got {diagram.ndim} dimensions.")

    features = {}
    for dim in homology_dimensions:
        # Filter the 2D diagram for the current dimension
        dim_diagram = diagram_2d[diagram_2d[:, 2] == dim]

        # Filter out infinite death times for lifespan calculation
        finite_diagram = dim_diagram[dim_diagram[:, 1] != np.inf]

        if finite_diagram.shape[0] > 0:
            births = finite_diagram[:, 0]
            deaths = finite_diagram[:, 1]
            lifespans = deaths - births

            features[f'h{dim}_feature_count'] = float(len(lifespans))
            features[f'h{dim}_avg_lifespan'] = np.mean(lifespans)
            features[f'h{dim}_max_lifespan'] = np.max(lifespans)
            features[f'h{dim}_std_lifespan'] = np.std(lifespans)
            features[f'h{dim}_avg_birth'] = np.mean(births)
            features[f'h{dim}_avg_death'] = np.mean(deaths)
        else:
            # If no finite features exist for this dimension, fill with zeros
            features[f'h{dim}_feature_count'] = 0.0
            features[f'h{dim}_avg_lifespan'] = 0.0
            features[f'h{dim}_max_lifespan'] = 0.0
            features[f'h{dim}_std_lifespan'] = 0.0
            features[f'h{dim}_avg_birth'] = 0.0
            features[f'h{dim}_avg_death'] = 0.0

    if verbose:
        print("--- Extracted Features ---")
        if features:
            max_key_length = max(len(key) for key in features.keys())
            for key, value in features.items():
                print(f"\t{key:<{max_key_length}}: {value:.4f}")
        else:
            print("\tNo features were extracted.")

    return features

def sample_time_series_slices(
    time_series: pd.Series,
    slice_length: float,
    n_slices: Optional[int] = None,
    gap: Optional[float] = None,
    arbitrary: bool = False,
    random_seed: Optional[int] = None,
) -> List[pd.Series]:
    """
    Samples slices from a time series, either systematically or arbitrarily.

    Args:
        time_series (pd.Series): The input time series data, with a time index.
        slice_length (float): The desired length of each slice in seconds.
        n_slices (Optional[int]): The number of slices to extract.
            - For systematic sampling, slices are distributed evenly.
            - For arbitrary sampling, this many random slices are chosen.
            - Must be provided if 'arbitrary' is True.
        gap (Optional[float]): The time gap in seconds between the end of one
            slice and the start of the next. Used only for systematic sampling.
            Cannot be used with 'n_slices'.
        arbitrary (bool): If True, slices are chosen randomly (but without
            overlap). If False (default), slices are chosen systematically.
        random_seed (Optional[int]): A seed for the random number generator
            to ensure reproducibility of arbitrary sampling.

    Returns:
        List[pd.Series]: A list containing the extracted time series slices.

    Raises:
        ValueError: If input parameters are invalid or conflicting.
        TypeError: If 'time_series' is not a pandas Series with a time index.
    """
    if not isinstance(time_series, pd.Series) or not isinstance(time_series.index, pd.Index):
        raise TypeError("Input 'time_series' must be a pandas Series with a time index.")

    # --- Calculate Sampling Frequency and Convert Seconds to Points ---
    if len(time_series.index) > 1:
        fs = 1 / (time_series.index[1] - time_series.index[0])
    else:
        fs = 1.0 # Assume fs of 1.0 if only one point
        
    slice_length_pts = int(slice_length * fs)
    gap_pts = int(gap * fs) if gap is not None else None

    total_len = len(time_series)
    if not (0 < slice_length_pts <= total_len):
        raise ValueError(
            f"slice_length ({slice_length}s) results in a slice length ({slice_length_pts} points) "
            f"that is not positive or exceeds the time series length ({total_len} points)."
        )

    # --- Parameter Validation ---
    if arbitrary:
        if gap is not None:
            raise ValueError("Cannot specify 'gap' when 'arbitrary' is True.")
        if n_slices is None:
            raise ValueError("Must specify 'n_slices' when 'arbitrary' is True.")
    else:  # Systematic
        if n_slices is not None and gap is not None:
            raise ValueError("Cannot specify both 'n_slices' and 'gap' for systematic sampling.")
        if n_slices is None and gap is None:
            raise ValueError("Must specify either 'n_slices' or 'gap' for systematic sampling.")

    start_indices = []
    
    # --- Slicing Logic ---
    if arbitrary:
        print("--- Performing Arbitrary Slicing ---")
        rng = np.random.default_rng(random_seed)
        
        possible_starts = list(range(total_len - slice_length_pts + 1))
        
        max_possible_slices = total_len // slice_length_pts
        if n_slices > max_possible_slices:
            print(
                f"Warning: Cannot fit {n_slices} non-overlapping slices. "
                f"Reducing to the maximum possible: {max_possible_slices}."
            )
            n_slices = max_possible_slices

        for _ in range(n_slices):
            if not possible_starts:
                print("Warning: Ran out of possible locations for slices. Generated fewer slices than requested.")
                break
            
            start = rng.choice(possible_starts)
            start_indices.append(start)
            
            forbidden_start = start - slice_length_pts + 1
            forbidden_end = start + slice_length_pts - 1
            possible_starts = [
                idx for idx in possible_starts 
                if not (forbidden_start <= idx <= forbidden_end)
            ]
        start_indices.sort()

    else: # Systematic
        print("--- Performing Systematic Slicing ---")
        if n_slices is not None:
            if n_slices * slice_length_pts > total_len:
                new_n_slices = total_len // slice_length_pts
                print(
                    f"Warning: Cannot fit {n_slices} slices of length {slice_length}s. "
                    f"Reducing to {new_n_slices} evenly spaced slices."
                )
                n_slices = new_n_slices
            
            if n_slices > 0:
                start_indices = np.linspace(0, total_len - slice_length_pts, n_slices, dtype=int)

        elif gap_pts is not None:
            if total_len >= slice_length_pts:
                num_slices = (total_len - slice_length_pts) // (slice_length_pts + gap_pts) + 1
                start_indices = [i * (slice_length_pts + gap_pts) for i in range(num_slices)]

    if not list(start_indices):
        print("Warning: No slices were generated with the given parameters.")
        return []

    # --- Create Slices and Print Summary ---
    slices = [time_series.iloc[i : i + slice_length_pts] for i in start_indices]

    print("\n--- Time Series Slicing Summary ---")
    print(f"Number of slices generated: {len(slices)}")
    print(f"Length of each slice: {slice_length}s ({slice_length_pts} points at {fs:.2f} Hz)")
    if slices:
        print("Time intervals of slices:")
        for i, s in enumerate(slices):
            if not s.empty:
                start_time = s.index[0]
                end_time = s.index[-1]
                print(f"  - Slice {i+1}: {start_time:.4f}s to {end_time:.4f}s")

    return slices
