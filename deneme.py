from gtda.time_series import SingleTakensEmbedding
import numpy as np
import os

# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import plotly.io as pio
from gtda.homology import VietorisRipsPersistence

# --- Set Plotting Themes to Light Mode ---
pio.templates.default = "plotly_white" # For plotly and giotto-tda plots
plt.style.use('default') # For matplotlib plots

print("All imports are done!")

# Define the non-periodic example with a higher sampling rate
x_nonperiodic = np.linspace(0, 100, 2000)
y_nonperiodic = np.cos(x_nonperiodic) + np.cos(np.pi * x_nonperiodic)

print("The time series simulated!")

# The helper function to display the optimal search parameters
def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )

    return y_embedded


max_embedding_dimension = 30
max_time_delay = 30
stride = 8

embedder_nonperiodic = SingleTakensEmbedding(
    parameters_type="search",
    n_jobs=-1,
    time_delay=max_time_delay,
    stride=stride,
)

y_nonperiodic_embedded = fit_embedder(embedder_nonperiodic, y_nonperiodic)

print("The time series is embedded!")


y_nonperiodic_embedded = y_nonperiodic_embedded[None, :, :]

print("The shape of the data is made suitable for the VR Persistence")