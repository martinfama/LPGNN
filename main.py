from operator import attrgetter
import numpy as np
import networkx as nx

import torch
import torch_geometric as pyg

import LPGNN

import matplotlib.pyplot as plt

## Plot results, and log

LPGNN.FullRun.FullRun(graph_type='PS', N=600, avg_k=5, gamma=2.5, Temp=0.0, seed=100, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=1000, 
                 attributes=['degree_centrality', 'betweenness_centrality'], save_name='default')

# LPGNN.FullRun.FullRun(graph_type='Barabasi', N=300, M=4, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=20, 
#                 attributes=['degree_centrality', 'betweenness_centrality', 'closeness_centrality'], save_name='default')

# LPGNN.FullRun.FullRun(graph_type='Erdos-Renyi', N=300, p=0.05, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=20,
#                 attributes=['degree_centrality', 'betweenness_centrality', 'closeness_centrality'], save_name='default')