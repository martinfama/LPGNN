import torch_geometric.transforms as pyg_T
import torch_geometric.data as pyg_data
import torch_geometric as pyg
import torch
from . import network_generator as ng
import networkx as nx
import numpy as np

def getPSNetwork(N, avg_k, gamma, Temp, seed):
    PS_nx, PS = ng.generatePSNetwork_nx(N=N, avg_k=avg_k, gamma=gamma, Temp=Temp, seed=seed)
    data = pyg.utils.from_networkx(PS_nx)
    data.x = torch.Tensor(np.transpose([list(nx.degree_centrality(PS_nx).values()), 
                                        list(nx.betweenness_centrality(PS_nx).values())]))
    return PS, PS_nx, data

def TrainTestSplit(data, test_ratio, val_ratio, neg_samples=False):
    # Split data into train and test sets
    RLS = pyg_T.RandomLinkSplit(is_undirected=True, 
                                num_val=val_ratio, num_test=test_ratio,
                                add_negative_train_samples=neg_samples,
                                split_labels=True)
    train_data, val_data, test_data = RLS(data)
    return train_data, val_data, test_data