"""
Utility functions for quantile prediction
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license


import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def build_alpha_quantiles_map(target_alpha: List, target_quantiles: List):
    """
    Build the map between PIs coverage levels and related quantiles
    """
    alpha_q = {'med': target_quantiles.index(0.5)}
    for alpha in target_alpha:
        alpha_q[alpha] = {
            'l': target_quantiles.index(alpha / 2),
            'u': target_quantiles.index(1 - alpha / 2),
        }
    return alpha_q


def fix_quantile_crossing(preds: np.array):
    """
    Fix crossing in the predicted quantiles by means of post-hoc sorting
    """
    return np.sort(preds, axis=-1)


def plot_quantiles(results: pd.DataFrame, target: str, path_to_save: str = None):
    """
    Plot predicted quantiles
    """
    title = target
    idx = results[target].index
    fig1, ax1 = plt.subplots()

    # Get the list of columns
    columns = results.columns.to_list()
    half = len(columns) // 2

    # Create a colormap
    colormap = cm.get_cmap('viridis', half)

    # Plot the first half of the columns with a gradient of colors
    for i in range(half):
        ax1.plot(idx, results[columns[i]], linestyle="-", color=colormap(i), linewidth=0.9)

    # Plot the second half of the columns with the same colors but in opposite order
    for i in range(half, len(columns)):
        ax1.plot(idx, results[columns[i]], linestyle="-", color=colormap(half - 1 - (i - half)), linewidth=0.9)

    ax1.plot(idx, results[target], '-', color='firebrick', label='$y_{true}$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel("Predicted quantiles")
    ax1.set_title(title)
    fig1.show()

    if path_to_save is not None:
        fig1.savefig(path_to_save)
