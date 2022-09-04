import networkx as nx
import torch as th
import torch_geometric as pyg

def minimum_spanning_tree(data:pyg.data.Data, edge_index='edge_index'):
    """ Function to get the minimum spanning tree of a PyTorch-Geometric Data object.

    Args:
        data (pyg.data.Data): The network to be pruned.

    Returns:
        (pyg.data.Data): The pruned network.
    """

    d = data.clone()
    d.edge_index = getattr(data, edge_index)
    G = pyg.utils.to_networkx(data, to_undirected=True)
    # apply weight to edges
    for u, v in G.edges():
        G[u][v]['weight'] = (list(nx.adamic_adar_index(G, [(u,v)]))[0][2]+0.1)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst_pyg = data.clone()
    setattr(mst_pyg, edge_index, pyg.utils.to_undirected(th.Tensor(list(mst.edges())).long().T))
    return mst_pyg