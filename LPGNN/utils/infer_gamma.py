import powerlaw
import torch_geometric as pyg

def infer_gamma(data:pyg.data.Data, **kwargs):
    """ Infers the distribution scale factor "gamma" of a graph's degree
        distribution. Uses the module ``powerlaw``.

    Args:
        data (pyg.data.Data): The graph to calculate the scale factor of.

    Returns:
        powerlaw.Fit: The calculated gamma and amplitude factor of powerlaw.
    """

    fit = powerlaw.Fit(pyg.utils.degree(data.edge_index[0]), discrete=True, verbose=False, **kwargs)
    return fit