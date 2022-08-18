import torch as th
import torch_geometric as pyg

import igraph
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.optimize
import pandas as pd
import copy
import time
from .metrics import *

# #infer some of the network's properties, such as the average node degree
# #and the average clustering coefficient
# def infer_avg_k(self, inplace=False):
#     avg_k = np.average(self.graph.degree())
#     if inplace: self.avg_k = avg_k
#     return avg_k

# def infer_clustering_coefficient(self, inplace=False):
#     clustering_coeff = self.graph.transitivity_avglocal_undirected()
#     if inplace: self.clustering_coeff = clustering_coeff
#     return clustering_coeff

# def infer_gamma(self, inplace=False):
#     gamma = igraph.power_law_fit(data=self.graph.degree()).alpha
#     if inplace: self.gamma = gamma
#     return gamma

#generate a network according to the PS model       #seed is for randomness
def generatePSNetwork(N=0, avg_k=0, gamma=0, T=0, seed=0):

    th.manual_seed(seed)

    data = pyg.data.Data(is_directed=False)
    data.is_PS = True
    data.num_nodes = N
    data.avg_k = avg_k
    data.gamma = gamma
    data.T = T
    data.seed = seed
    # Initialize node positions as 2D torch tensor (r, theta). Both are zero
    data.node_positions = th.zeros(size=(data.num_nodes,2))
    # Set node angular positions to random uniform distribution [0, 2π). We
    # can do this now because it doesn't affect PS algorithm.
    data.node_positions[:,1] = th.rand(size=(data.num_nodes,))*2*th.pi
    # We _can't_ set node radial positions now because with popularity fading,
    # this coordinate changes as time progresses in the network generation.

    # beta controls popularity fading
    beta = 1/(gamma-1)
    # avg number of connections
    m = round(avg_k/2)
    # since the nodes 0,1,2,...,m will be fully connected, we connect them now
    data.edge_index = th.Tensor([[i, j] for i in th.arange(0, m+1) for j in th.arange(i+1, m+1)]).T.type(th.int64)
    # duplicate all connections but switched (i.e. (a,b) -> (b,a)), to make undirected graph
    data.edge_index = pyg.utils.to_undirected(data.edge_index)

    for t in range(m+1, N):

        # radial position of node entering network
        r_t = np.log(t+1)

        # update all of the existing nodes radial coordinates according to
        # r_s(t) = beta*r_s + (1-beta)*r_t, w/ r_s = ln(s), r_t = ln(t)
        # which simulates popularity decay due to aging
        r_s = th.log(th.arange(1, t+1))
        data.node_positions[:t,0] = 2*(1-beta)*r_t + 2*beta*r_s
        data.node_positions[t,0]  = 2*r_t

        #R_t is the radius of the circle containing the network in Euclidean space,
        #which in turn contains the entirety of the Hyperbolic space
        R_t = 0
        # If beta == 1, popularity fading is not simulated
        if beta == 1:
            R_t = 2*r_t
        elif beta < 1 and T == 0:
            R_t = 2*r_t - 2*np.log((2*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.pi*m*(1 - beta))) #SI, Eq. 10
        else:
            #R_t = 2*r_t - 2*np.log((2*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.pi*m*(1 - beta))) #SI, Eq. 10
            R_t = 2*r_t - 2*np.log((2*T*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.sin(T*np.pi)*m*(1 - beta))) #SI, Eq. 26
        
        #save all hyperbolic distances between new node and other nodes
        d = hyperbolic_distances(data.node_positions[:t-1], data.node_positions[t]).sort()
        
        #If T = 0, simply connect to the m hyperbolically closest nodes
        if T == 0:
            new_edges = th.stack([d.indices[:m], th.empty(m).fill_(t)])
            data.edge_index = th.cat((data.edge_index, new_edges), dim=1)
            data.edge_index = pyg.utils.to_undirected(data.edge_index)
        
        else:
            # probability that the new node connects to the other nodes in the network
            # also, sort this list, which returns both the sorted list and the sorted indices
            #     values: p.values
            #    indices: p.indices
            p = 1 / (1 + th.exp((d.values - R_t)/(2*T)))
            # get m nodes to connect to, sampled by the probabilities given by p.values
            selected_nodes = np.random.choice(d.indices.detach().numpy(), size=m, p=(p/th.sum(p)).detach().numpy(), replace=False)
            new_edges = th.stack([selected_nodes, th.empty(m).fill_(t)])
            data.edge_index = th.cat((data.edge_index, new_edges), dim=1)
            data.edge_index = pyg.utils.to_undirected(data.edge_index)
    
    data.edge_index = data.edge_index.type(th.int64)

    return data


def drawPSNetwork(PS:pyg.data.Data):
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
    PS_nx = pyg.utils.to_networkx(PS, to_undirected=True)
    
    degrees = pyg.utils.degree(PS.edge_index[0]).detach().numpy()
    max_d = np.max(degrees)
    min_d = np.min(degrees)

    node_size = 2*((1000/(1+np.exp( -((degrees-min_d)/(max_d-min_d)-0.6)*0.5 ))) + 10).astype(np.int64)

    nx.draw(PS_nx, ax=ax, pos=dict(zip(range(PS.num_nodes), np.flip(PS.node_positions.detach().numpy(), axis=1))), 
                          node_color=PS.node_positions[:,1].detach().numpy(), cmap=plt.cm.rainbow,
                          node_size=node_size)
    
    return fig, ax
    