import torch_geometric.transforms as pyg_T
import torch_geometric.data as pyg_data
import torch_geometric as pyg
import torch
from . import popularity_similarity as pop_sim
import networkx as nx
import igraph as ig
import numpy as np

def add_self_loops_to_graph(G):
    """Add self-loops to a graph

    Args:
        G (nx.Graph or ig.Graph): Graph to add self-loops to.

    Returns:
        nx.Graph or ig.Graph: Graph with self-loops added.
    """
    if type(G) == nx.Graph:
        for node in G.nodes():
            G.add_edge(node, node)
    elif type(G) == ig.Graph:
        for node in G.vs:
            G.add_edge(node.index, node.index)
    return G

def get_ps_network(N, avg_k, gamma, Temp, seed):
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
    PS_nx = add_self_loops_to_graph(PS_nx)
    PS = add_self_loops_to_graph(PS)
    data = pyg.utils.from_networkx(PS_nx)
    return PS, PS_nx, data

def get_barabasi_network(N, M):
    """Generate a Barabasi-Albert network

    Args:
        N (int): Number of nodes
        M (int): Number of connections new nodes should establish
    Returns:
    tuple: A tuple containing the three network types: (igraph, networkx, torch).
    """
    G_nx, G = ng.generateBarabasiNetwork(N=N, M=M)
    data = pyg.utils.from_networkx(G_nx)
    return G, G_nx, data

def get_erdos_renyi_network(N, p):
    """Generate a Erdos-Renyi network

    Args:
        N (int): Number of nodes
        p (float): Probability of link creation
    Returns:
    tuple: A tuple containing the three network types: (igraph, networkx, torch).
    """
    G_nx, G = ng.generateErdos_RenyiNetwork(N=N, p=p)
    data = pyg.utils.from_networkx(G_nx)
    return G, G_nx, data

def train_test_split(data, test_ratio, val_ratio, neg_samples=False):
    """Split a data graph object (``torch.Tensor``) into train, test and validation sets of edges.

    Args:
        data (pyg.data.Data): Full graph data.
        test_ratio (_type_): Size of test set as a fraction of the total size of graph.
        val_ratio (_type_): Size of validation set as a fraction of the total size of graph.
        neg_samples (bool, optional): Whether to include negative samples (i.e. non existent edges). 
                                      Defaults to False.

    Returns:
        data_c (pyg.data.Data): A copy of the input data object with the train, test and validation masks.
    """
    data_c = data.clone()
    RLS = pyg_T.RandomLinkSplit(is_undirected=True, 
                                num_val=val_ratio, num_test=test_ratio,
                                add_negative_train_samples=neg_samples,
                                split_labels=True)
    train_data, val_data, test_data = RLS(data_c)
    data_c.train_pos_edge_label = train_data.pos_edge_label
    data_c.train_pos_edge_label_index = pyg.utils.to_undirected(train_data.pos_edge_label_index)
    if neg_samples:
        data_c.train_neg_edge_label = train_data.neg_edge_label
        data_c.train_neg_edge_label_index = pyg.utils.to_undirected(train_data.neg_edge_label_index)
    if val_ratio > 0:
        data_c.val_pos_edge_label = val_data.pos_edge_label
        data_c.val_pos_edge_label_index = pyg.utils.to_undirected(val_data.pos_edge_label_index)
        if neg_samples:
            data_c.val_neg_edge_label = val_data.neg_edge_label
            data_c.val_neg_edge_label_index = pyg.utils.to_undirected(val_data.neg_edge_label_index)
    if test_ratio > 0:
        data_c.test_pos_edge_label = test_data.pos_edge_label
        data_c.test_pos_edge_label_index = test_data.pos_edge_label_index
        if neg_samples:
            data_c.test_neg_edge_label = test_data.neg_edge_label
            data_c.test_neg_edge_label_index = test_data.neg_edge_label_index
    return data_c