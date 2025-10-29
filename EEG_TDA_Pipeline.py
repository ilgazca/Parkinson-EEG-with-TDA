'''
This script is intended to automize the process of 
1- reading a .mat file
2- turning it into a time series (in terms of a pd.dataframe)
3- downsample it
4- apply taken's embedding
5- calculate persistent homology of the embedded series
6- extract some features 
'''

#--- Imports ---
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from eeg_utils import mat_to_dataframe, downsample_eeg_dataframe


#--- Load the data ---
result = mat_to_dataframe("sub-0cGdk9_HoldL_MedOff_run1_LFP_Hilbert/sub_AbzsOg_HoldL_MedOff_merged_LFP_Hilbert.mat")

# Unpack the results if the file was processed successfully
if result:
    df, left_lfp, right_lfp = result

    #--- Plot the data ---
    print("\nPlotting first 5000 samples...")
    plot_slice = 5000

    df.iloc[:plot_slice].plot(
        subplots=True,  # Plot each channel separately
        layout=(2, 1),  # Arrange in 2 rows, 1 column
        grid=True,
        title="LFP Time Series (First 5000 Samples)",
        figsize=(15, 6)  # Width, Height in inches
    )
    plt.xlabel(df.index.name)
    plt.tight_layout()
    plt.show()


# --- Downsample and plot the data ---
target_fs = 50
df_dwn = downsample_eeg_dataframe(df=df,
                                  original_fs=2000,
                                  target_fs=target_fs,
                                  plot=False)

print("\nPlotting first 5000 samples...")
plot_slice = 125

df_dwn.iloc[:plot_slice].plot(
    subplots=True,   # Plot each channel separately
    layout=(2, 1),   # Arrange in 2 rows, 1 column
    grid=True,
    title=f"LFP Time Series (First {plot_slice} Samples)",
    figsize=(15, 6)  # Width, Height in inches
)
plt.xlabel(df_dwn.index.name)
plt.tight_layout()
plt.show()


num_of_samples_to_keep = 125

left = df_dwn['LFP-left-56'].iloc[:num_of_samples_to_keep]
right = df_dwn['LFP-right-34'].iloc[:num_of_samples_to_keep]


# --- TDA Magic ---
from gtda.time_series import SingleTakensEmbedding
from gtda.time_series import TakensEmbedding
import itertools

# Plotting Libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy

from eeg_utils import fit_embedder, filter_persistence_diagram, extract_features

# Set Plotting Themes to Light Mode
pio.templates.default = "plotly_white" # For plotly and giotto-tda plots
plt.style.use('default') # For matplotlib plots


# --- Taken's Embedding ---

max_embedding_dim = 30
max_time_delay = 30
stride = 1

embedder = SingleTakensEmbedding(
    parameters_type="search",
    time_delay=max_time_delay,
    dimension=max_embedding_dim,
    stride=stride,
    n_jobs=-1
)

left_embedded = fit_embedder(embedder, left)
right_embedded = fit_embedder(embedder, right)

left_embedded = left_embedded[None, :, :]
right_embedded = right_embedded[None, :, :]

homology_dims = [0, 1, 2, 3]

lifespan_threshold = 0.2

left_persistence = VietorisRipsPersistence(homology_dimensions=homology_dims, n_jobs=-1)
left_diagram = left_persistence.fit_transform(left_embedded)

left_diagram_flt = filter_persistence_diagram(left_diagram, lifespan_threshold=lifespan_threshold)
left_persistence.plot(left_diagram_flt)

right_persistence = VietorisRipsPersistence(homology_dimensions=homology_dims, n_jobs=-1)
right_diagram = right_persistence.fit_transform(right_embedded)

right_diagram_flt = filter_persistence_diagram(right_diagram, lifespan_threshold=lifespan_threshold)
right_persistence.plot(right_diagram)

# --- Extracting Features ---

PE = PersistenceEntropy()
left_pe_features = PE.fit_transform(left_diagram_flt)
print(left_pe_features)

PE = PersistenceEntropy()
right_pe_features = PE.fit_transform(right_diagram_flt)
print(right_pe_features)


left_sm_features = extract_features(left_diagram_flt, homology_dimensions=homology_dims, verbose=True)
right_sm_features = extract_features(right_diagram_flt, homology_dimensions=homology_dims, verbose=True)







