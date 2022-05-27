import optparse
import numpy as np
import LPGNN.network_analysis as na
import networkx as nx

import torch
import torch_geometric as pyg

import LPGNN.visualization.loss_plotting as loss_plotting
import LPGNN.DataSetup as DataSetup
import LPGNN.GraphNeuralNet as GraphNeuralNet
import LPGNN.LinkPrediction as LinkPrediction
import LPGNN.Logger as Logger

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

        'graph_type': 'PS',

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
    if options['graph_type'] == 'PS':
        G, G_nx, data = DataSetup.get_ps_network(N=options['N'], avg_k=options['avg_k'], gamma=options['gamma'], Temp=options['Temp'], seed=options['seed'])
    elif options['graph_type'] == 'Barabasi':
        G, G_nx, data = DataSetup.get_barabasi_network(N=options['N'], M=options['M'])
    elif options['graph_type'] == 'Erdos-Renyi':
        G, G_nx, data = DataSetup.get_erdos_renyi_network(N=options['N'], p=options['p'])

    ## Attributes to include in data.x (if any), for the GNN's to use
    if options['attributes'] is not None:
        _ = []
        if 'degree' in options['attributes']:
            #_.append( NormalizeArray(list(nx.degree(G_nx).values())) )
            _.append( list(nx.degree(G_nx).values()) )
        if 'degree_centrality' in options['attributes']:
            #_.append( NormalizeArray(list(nx.degree_centrality(G_nx).values())) )
            _.append( list(nx.degree_centrality(G_nx).values()) )
        if 'betweenness_centrality' in options['attributes']:
            #_.append( NormalizeArray(list(nx.betweenness_centrality(G_nx).values())) )
            _.append( list(nx.betweenness_centrality(G_nx).values()) )
        if 'closeness_centrality' in options['attributes']:
            #_.append( NormalizeArray(list(nx.closeness_centrality(G_nx).values())) )
            _.append( list(nx.closeness_centrality(G_nx).values()) )

        data.x = torch.tensor(np.transpose(_), dtype=torch.float)
        
    ## Split the data into train, test and validation sets
    train, val, test = DataSetup.train_test_split(data, test_ratio=0.1, val_ratio=0.1, neg_samples=True)

    PR_list = []
    for model in options['models']:
        if model == 'LaBNE':
            PR_list.append(GraphNeuralNet.LaBNE(G, test))
        if model == 'CN':
            PR_list.append(GraphNeuralNet.CN(G, test))
        if model == 'GraphSAGE':
            PR_list.append(GraphNeuralNet.GraphSAGE(data, train, test, val, options['epochs']))
            #loss_plotting.plot_losses_vs_epochs(PR_list[-1])
        if model == 'PNA':
            PR_list.append(GraphNeuralNet.PNA(data, train, test, val, options['epochs']))
            #loss_plotting.plot_losses_vs_epochs(PR_list[-1])
    plt.show()

    ## Plot the results and save them
    if options['save_name'] == 'default':
        save_name = f'{options["graph_type"]}_{options["N"]}'
    else:
        save_name = options['save_name']
    
    LinkPrediction.plot_pr_curves(PR_list=PR_list, save_name=save_name)

    return