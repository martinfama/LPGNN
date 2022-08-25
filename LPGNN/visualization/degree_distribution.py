import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch_geometric as pyg
from ..utils import infer_gamma

def powerlaw(k, A, b):
    return A*k**(-b)

def plot_degree_distribution(data:pyg.data.Data, log_scale=False, fit_powerlaw=False, **kwargs):
    """ Plots the degree distribution of a torch_geometric Data graph

    Args:
        data (pyg.data.Data): The graph to get the degree distribution from.

    Returns:
        plt.fig: Figure of plot
    """

    degrees = pyg.utils.degree(data.edge_index[0])

    fig, ax = plt.subplots(figsize=(10,7))
    if log_scale: ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P(k)$')
    ax.grid(alpha=0.5)

    counts, bin_edges = np.histogram(degrees, bins=np.logspace(np.log10(degrees.min()), np.log10(degrees.max()), 20), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    ax.scatter(bin_centers, counts)

    if fit_powerlaw:
        fit = infer_gamma(data)
        #k_space = np.logspace(np.log10(degrees.min()), np.log10(degrees.max()), fit.power_law.pdf().shape[0])
        k_space = np.arange(fit.power_law.xmin, fit.power_law.xmin+fit.power_law.pdf().shape[0])
        # ax.plot(k_space, powerlaw(k_space, fit.power_law.alpha-1, fit.power_law.alpha))
        ax.plot(k_space, fit.power_law.pdf())

    return fig