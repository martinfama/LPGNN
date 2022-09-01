import networkx as nx
import torch as th
import torch_geometric as pyg

def minimum_spanning_tree(data:pyg.data.Data):
    """ Function to get the minimum spanning tree of a PyTorch-Geometric Data object.

    Args:
        data (pyg.data.Data): The network to be pruned.

    Returns:
        (pyg.data.Data): The pruned network.
    """

    G = pyg.utils.to_networkx(data, to_undirected=True)
    # apply weight to edges
    for u, v in G.edges():
        G[u][v]['weight'] = 1/(G.degree(u) + G.degree(v))
    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst = pyg.utils.from_networkx(mst)
    return mst