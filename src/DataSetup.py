import torch_geometric.transforms as pyg_T
import torch_geometric.data as pyg_data
import torch_geometric as pyg
import torch
from . import network_generator as ng
import networkx as nx
import numpy as np

def getPSNetwork(N, avg_k, gamma, Temp, seed):
    """Generate a PS network using the network_generator module. For now, we will use
    three copies of the same network in types ``igraph.Graph``, ``networkx.Graph`` and
    ``torch.Tensor``, since the newer stuff works with networkx and torch, but the older
    LaBNE stuff works with igraph. TODO: Stop using igraph!!

    Args:
        N (int): Nodes in network
        avg_k (double): Average node degree
        gamma (double): Scaling factor for node degree
        Temp (double): Temperature for edge probabilities
        seed (int): Seed for random number generator

    Returns:
        tuple: A tuple containing the three network types: (igraph, networkx, torch).
    """
    PS_nx, PS = ng.generatePSNetwork_nx(N=N, avg_k=avg_k, gamma=gamma, Temp=Temp, seed=seed)
    data = pyg.utils.from_networkx(PS_nx)
    data.x = torch.Tensor(np.transpose([list(nx.degree_centrality(PS_nx).values()), 
                                        list(nx.betweenness_centrality(PS_nx).values())]))
    return PS, PS_nx, data

def TrainTestSplit(data, test_ratio, val_ratio, neg_samples=False):
    """Split a data graph object (``torch.Tensor``) into train, test and validation sets of edges.

    Args:
        data (torch.Tensor): Full graph data.
        test_ratio (_type_): Size of test set as a fraction of the total size of graph.
        val_ratio (_type_): Size of validation set as a fraction of the total size of graph.
        neg_samples (bool, optional): Whether to include negative samples (i.e. non existent edges). 
                                      Defaults to False.

    Returns:
        tuple: A tuple containing the three network subsets: (train_data, test_data, val_data).
    """
    RLS = pyg_T.RandomLinkSplit(is_undirected=True, 
                                num_val=val_ratio, num_test=test_ratio,
                                add_negative_train_samples=neg_samples,
                                split_labels=True)
    train_data, val_data, test_data = RLS(data)
    return train_data, val_data, test_data