import numpy as np
from src import network_analysis as na
import networkx as nx

import torch
import torch_geometric as pyg

from src import DataSetup
from src import GraphNeuralNet
from src import LinkPrediction
from src import Logger

import matplotlib.pyplot as plt

def NormalizeArray(v):
    """ Normalize an array """
    norm = np.linalg.norm(v)
    if norm == 0:
        norm=np.finfo(v.dtype).eps
    return v / norm

def FullRun(**kargs):
    """Run a complete test, including training, testing and logging. Should accept network characteristics,
    and models to use. 
    """

    options = {
        'N': 400,
        'avg_k': 6,
        'gamma': 2.4,
        'Temp': 0.1,
        'seed': 100,

        'models': [],
        'epochs': 100,

        'attributes': None,

        'save_name': 'test',
    }

    options.update(kargs)
    
    ## Generate a PS network. For now, we will use the three type of networks: igraph, networkx and PyG
    PS, PS_nx, data = DataSetup.getPSNetwork(N=options['N'], avg_k=options['avg_k'], gamma=options['gamma'], Temp=options['Temp'], seed=options['seed'])
    ## Attributes to include in data.x (if any), for the GNN's to use
    if options['attributes'] is not None:
        _ = []
        if 'degree' in options['attributes']:
            _.append( NormalizeArray(list(nx.degree(PS_nx).values())) )
        if 'degree_centrality' in options['attributes']:
            _.append( NormalizeArray(list(nx.degree_centrality(PS_nx).values())) )
        if 'betweenness_centrality' in options['attributes']:
            _.append( NormalizeArray(list(nx.betweenness_centrality(PS_nx).values())) )
        if 'closeness_centrality' in options['attributes']:
            _.append( NormalizeArray(list(nx.closeness_centrality(PS_nx).values())) )

        data.x = torch.tensor(np.transpose(_), dtype=torch.float)
    ## Split the data into train, test and validation sets
    train, val, test = DataSetup.TrainTestSplit(data, test_ratio=0.1, val_ratio=0.1, neg_samples=True)

    PR_list = []
    for model in options['models']:
        if model == 'LaBNE':
            PR_list.append(GraphNeuralNet.LaBNE(PS, test))
        if model == 'GraphSAGE':
            PR_list.append(GraphNeuralNet.GraphSAGE(data, train, test, val, options['epochs']))
        if model == 'PNA':
            PR_list.append(GraphNeuralNet.PNA(data, train, test, val, options['epochs']))

    ## Plot the results and save them
    if options['save_name'] == 'default':
        save_name = f'{options["N"]}_{options["avg_k"]}_{options["gamma"]}_{options["Temp"]}_{options["seed"]}'
    else:
        save_name = options['save_name']
    LinkPrediction.PlotPRCurves(PR_list=PR_list, save_name=save_name)

    return