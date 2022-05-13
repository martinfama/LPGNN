from operator import attrgetter
import numpy as np
from src import network_analysis as na
import networkx as nx

import torch
import torch_geometric as pyg

from src import DataSetup
from src import GraphNeuralNet
from src import LinkPrediction
from src import Logger
from src import FullRun

import matplotlib.pyplot as plt

## Plot results, and log

FullRun.FullRun(graph_type='PS', N=500, avg_k=6, gamma=2.5, Temp=0.0, seed=100, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=70, 
                 attributes=['degree_centrality'], save_name='default')

# FullRun.FullRun(graph_type='Barabasi', N=300, M=4, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=20, 
#                 attributes=['degree_centrality', 'betweenness_centrality', 'closeness_centrality'], save_name='default')

# FullRun.FullRun(graph_type='Erdos-Renyi', N=300, p=0.05, models=['LaBNE', 'CN', 'GraphSAGE', 'PNA'], epochs=20,
#                 attributes=['degree_centrality', 'betweenness_centrality', 'closeness_centrality'], save_name='default')