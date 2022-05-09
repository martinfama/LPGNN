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
FullRun.FullRun(N=100, avg_k=10, gamma=2.2, Temp=0.1, seed=100, models=['LaBNE', 'GraphSAGE'], epochs=100, attributes=['degree_centrality'],
                save_name='default')