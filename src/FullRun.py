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

def LaBNE(PS, test):
    ## Generate LaBNE embedding and Precision-Recall on it
    PS_train = PS.copy()
    test_edges = np.transpose(test.pos_edge_label_index)
    PS_train.delete_edges([(x[0], x[1]) for x in test_edges])
    PR_LaBNE = na.precision_recall_snapshots(PS_train, PS, metric='LaBNE', step=1, plot=False)
    PR_LaBNE['label'] = 'LaBNE'
    return PR_LaBNE

def GraphSAGE(data, train, test, val, epochs, **kargs):
    ## GraphSAGE model and PR on it
    graphSAGE_model = pyg.nn.GraphSAGE(in_channels=data.num_features, hidden_channels=128, out_channels=32, num_layers=3)
    optimizer = torch.optim.SGD(graphSAGE_model.parameters(), lr=0.01)

    print(f'Training model: {graphSAGE_model} for {epochs} epochs')
    loss = GraphNeuralNet.train_model(model=graphSAGE_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
    print(f'Train loss: {loss}')
    R_SAGE, P_SAGE, predictions = LinkPrediction.PrecisionRecallTrainedModel(model=graphSAGE_model, train_data=train, test_data=test)
    _ = {'recall': R_SAGE, 'precision': P_SAGE, 'label': 'GraphSAGE'}
    return _

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

    for model in options['models']:
        if model == 'GraphSAGE':
            PR_SAGE = GraphSAGE(data, train, test, val, options['epochs'])
        if model == 'LaBNE':
            PR_LaBNE = LaBNE(PS, test)

    ## Plot the results and save them
    if options['save_name'] == 'default':
        save_name = f'{options["N"]}_{options["avg_k"]}_{options["gamma"]}_{options["Temp"]}_{options["seed"]}'
    else:
        save_name = options['save_name']
    LinkPrediction.PlotPRCurves(PR_list=[PR_SAGE, PR_LaBNE], save_name=save_name)

    return