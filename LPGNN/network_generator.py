__version__ = '1.2.3'

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

class Network(object):
    def __init__(self, **kwargs):
        
        self.graph = None
        # different ways the Networks graph can be created
        #receive a Graph() object. overrides 'edgelist'
        if 'graph' in kwargs:
            if type(kwargs.get('graph')) != igraph.Graph:
                raise TypeError('graph must be of type igraph.Graph()')
            self.graph = copy.deepcopy(kwargs.get('graph'))
        #receive an edgelist as a filename, and create a Graph() with that
        elif 'edgelist' in kwargs:
            #self.graph = igraph.Graph.Read_Ncol(kwargs.get('edgelist'), names=True, directed=False)
            self.graph = igraph.Graph.Read_Edgelist(kwargs.get('edgelist'), directed=False)
            self.graph.delete_vertices([0])

        #if a valid graph has been created... 
        if self.graph != None:
            #set self.N to the node count
            self.N = self.graph.vcount()
            
            #the try checks to see if the graph has an 'o_i' attribute, which keeps track of the original indexes
            #if it doesn't exist, give graph that attribute
            try:
                self.graph.vs['o_i']
            except:
                self.graph.vs['o_i'] = np.arange(self.N)

        # force set some of the graph's properties:
        # avg_k is the average node degree
        if 'avg_k' in kwargs:
            self.avg_k = kwargs.get('avg_k')
        # gamma is the scaling factor, assuming a scale free network
        if 'gamma' in kwargs:
            self.gamma = kwargs.get('gamma')
        # T is the network Terature, which is related to it's clustering coefficient \bar{c}
        if 'T' in kwargs:
            self.T = kwargs.get('T')
    
    def generatePSNetwork(self, N=None, avg_k=None, gamma=None, T=None, seed=None):
        
        if N != None:
            self.N = N
        if avg_k != None:
            self.avg_k = avg_k
        if gamma != None:
            self.gamma = gamma
        if T != None:
            self.T = T
        if seed != None:
            self.seed = time.time()
        
        self.graph = generatePSNetwork(N, avg_k, gamma, T, seed)

        self.is_PS_generated = True

    #infer some of the network's properties, such as the average node degree
    #and the average clustering coefficient
    def infer_avg_k(self, inplace=False):
        avg_k = np.average(self.graph.degree())
        if inplace: self.avg_k = avg_k
        return avg_k

    def infer_clustering_coefficient(self, inplace=False):
        clustering_coeff = self.graph.transitivity_avglocal_undirected()
        if inplace: self.clustering_coeff = clustering_coeff
        return clustering_coeff

    def infer_gamma(self, inplace=False):
        gamma = igraph.power_law_fit(data=self.graph.degree()).alpha
        if inplace: self.gamma = gamma
        return gamma

    # some metrics, such as normalized common neighbors

    # returns the common neighbors of nodes i and j, normalized by the geometric mean of their degrees: sqrt(degree(i) * degree(j))
    def common_neighbors(self, i=int, j=int):
        return len(set(self.graph.neighbors(i)).intersection(self.graph.neighbors(j)))/np.sqrt(self.graph.degree(i)*self.graph.degree(j))

#generate a network according to the PS model       #seed is for randomness
def generatePSNetwork(N=0, avg_k=0, gamma=0, T=0, seed=0):

    th.manual_seed(seed)

    data = pyg.data.Data(is_directed=False)
    data.num_nodes = N
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
            new_edges = th.stack([d.indices[:m], th.empty(m).fill_(m+1)])
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
            new_edges = th.stack([selected_nodes, th.empty(m).fill_(m+1)])
            data.edge_index = th.cat((data.edge_index, new_edges), dim=1)
            data.edge_index = pyg.utils.to_undirected(data.edge_index)

    return data

#generates the Laplacian Based Network Embedding for a given Network
def generateLaBNE(network=None, graph=None, eigenvector_k=3, scatterPlot=False, plotEdges=False):

    if network == None:
        if graph == None:
            print(f'No network or graph.')
            return -1
        else:
            network = Network(graph=graph)

    graph = network.graph
    N = network.N
    
    try:
        if network.avg_k != None:
            avg_k = network.avg_k
        else:
            raise Exception()
    except:
        avg_k = network.infer_avg_k()
    try:
        if network.gamma != None:
            gamma = network.gamma
        else:
            raise Exception()
    except:
        gamma = network.infer_gamma()


    #T = network.T
    beta = 1/(gamma-1)
    m = round(avg_k/2)

    #get Laplacian matrix of the graph (L = D - A). We pass it to a sparse matrix type supported by SciPy
    #so that we can use scipy's sparse linear algebra tools
    L = np.array(graph.laplacian(), dtype=np.float64)
    L = scipy.sparse.coo_matrix(L)
    #L = scipy.sparse.csr_matrix(L)
    #L = L.asfptype()
    #the tol parameter of scipy.sparse.lingalg() is to give an error tolerance. for smaller networks, convergence seems to be
    #guaranteed with high precision. for larger networks, convergence maybe a problem, so an error tolerance is given. criteria
    #the same as in Alanis-Lobato's NetHypGeom library
    if N < 10000:
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=eigenvector_k, which='LM', sigma=0, return_eigenvectors=True)
        except(RuntimeError):
            Levenberg_c = np.zeros(np.shape(L))
            np.fill_diagonal(Levenberg_c, 0.01)
            Levenberg_c = scipy.sparse.coo_matrix(Levenberg_c)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L+Levenberg_c, k=eigenvector_k, which='LM', sigma=0, return_eigenvectors=True)
        #eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=eigenvector_k, which='SM', return_eigenvectors=True)
    else:        
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=eigenvector_k, which='LM', sigma=0, return_eigenvectors=True, tol=1E-7, maxiter=5000)
        except(RuntimeError):
            Levenberg_c = np.zeros(np.shape(L))
            np.fill_diagonal(Levenberg_c, 0.01)
            Levenberg_c = scipy.sparse.coo_matrix(Levenberg_c)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L+Levenberg_c, k=eigenvector_k, which='LM', sigma=0, return_eigenvectors=True, tol=1E-7, maxiter=5000)

        #eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=eigenvector_k, which='SM', return_eigenvectors=True, tol=1E-7)
    eigenvectors = np.transpose(eigenvectors)
    x, y = eigenvectors[1].real, eigenvectors[2].real

    labne_graph = copy.deepcopy(network.graph)
    
    #sort original indexes of graph by degree. 
    #sorted_o_i = np.transpose(sorted(np.transpose([network.graph.vs['o_i'], network.graph.degree()]), key= lambda x: x[1], reverse=True))[0]
    #labne_graph.vs['o_i'] = sorted_o_i
    labne_graph.vs['degree_sorted_index'] = np.argsort(graph.degree())[::-1]

    for t in range(N):
        r_t = np.log(t+1)
        labne_graph.vs[labne_graph.vs[t]['degree_sorted_index']]['r'] = 2*beta*r_t + 2*(1-beta)*np.log(N)
        #labne_graph.vs[t]['r'] = 2*beta*r_t + 2*(1-beta)*np.log(N)
        labne_graph.vs[t]['theta'] = np.arctan2(y[t], x[t])
        if labne_graph.vs[t]['theta'] < 0:
            labne_graph.vs[t]['theta'] = labne_graph.vs[t]['theta']+2*np.pi

    #assign a rainbow palette based on the angular coordinate
    #if graph was generated by PS Method apply the original colors for comparison
    rainbow_palette = igraph.RainbowPalette(n=N)
    try:
        if network.is_PS_generated:
            for i in range(N):
                labne_graph.vs[i]['color'] = rainbow_palette.get(int(graph.vs[i]['theta']/(2*np.pi)*N))
        else:
            raise Exception()
    except:
        for i in range(N):
            labne_graph.vs[i]['color'] = rainbow_palette.get(int(labne_graph.vs[i]['theta']/(2*np.pi)*(N-1)))

    #give the igraph Graph x and y coordinates, so as to be able to plot it
    #and show the hyperbolic structure
    #labne_graph.vs['x'] = labne_graph.vs['r']*np.cos(labne_graph.vs['theta'])
    #labne_graph.vs['y'] = labne_graph.vs['r']*np.sin(labne_graph.vs['theta'])
    labne_graph.vs['y'] = eigenvectors[1]
    labne_graph.vs['x'] = eigenvectors[2]
    labne_graph.vs['size'] = node_size_by_degree(graph=labne_graph, m=1)

    #also return a matplotlib scatter plot with the coordinates given by the embedding
    #leaving this aside for a while, igraph already works well enough
    """
    if scatterPlot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

        ax1.grid(alpha=0.5)
        ax2.grid(alpha=0.5)

        if plotEdges:
            for i in range(N):
                for j in range(i, N):
                    if graph.are_connected(i, j):
                        ax1.plot([graph.vs[i]['r']*np.cos(graph.vs[i]['theta']), graph.vs[j]['r']*np.cos(graph.vs[j]['theta'])], 
                                 [graph.vs[i]['r']*np.sin(graph.vs[i]['theta']), graph.vs[j]['r']*np.sin(graph.vs[j]['theta'])],
                                linewidth=0.2,
                                c='black', zorder=-1)
                        ax2.plot([labne_graph.vs[i]['r']*np.cos(labne_graph.vs[i]['theta']), labne_graph.vs[j]['r']*np.cos(labne_graph.vs[j]['theta'])], 
                                 [labne_graph.vs[i]['r']*np.sin(labne_graph.vs[i]['theta']), labne_graph.vs[j]['r']*np.sin(labne_graph.vs[j]['theta'])],
                                linewidth=0.2,
                                c='black', zorder=-1)

        ax1.scatter(graph.vs['r']*np.cos(graph.vs['theta']), graph.vs['r']*np.sin(graph.vs['theta']),
                   s=16, c=graph.vs['theta'], cmap='rainbow', zorder=0)

        degree_sorted, theta_sorted = zip(*sorted(zip(graph.vs.degree(), graph.vs['theta'])))
        degree_sorted = degree_sorted[::-1]
        theta_sorted  = theta_sorted[::-1]

        ax2.scatter(labne_graph.vs['r']*np.cos(labne_graph.vs['theta']), labne_graph.vs['r']*np.sin(labne_graph.vs['theta']),
                    s=16, c=labne_graph.vs['theta'], cmap='rainbow', zorder=0, marker='^')
        
        #return labne_graph, labne_nodes, fig, ax1, ax2
        return labne_graph, fig, ax1, ax2
    """
    return labne_graph

#simple function to vary node size by degree
def node_size_by_degree(graph=None, m=1):
    return m*2*np.log(np.array(graph.degree())+np.full(graph.vcount(), 2))*np.log(np.e+100/graph.vcount())

def generatePSNetwork_nx(N=0, avg_k=0, gamma=0, T=0, seed=0):
    PS = generatePSNetwork(N=N, avg_k=avg_k, gamma=gamma, T=T, seed=seed)
    PS_nx = nx.Graph(PS.get_edgelist(), pos=np.transpose([PS.vs['x'], PS.vs['y']]), color=PS.vs['color'], size=PS.vs['size'])
    return PS_nx, PS

def generateBarabasiNetwork(N=0, M=0):
    G_nx = nx.barabasi_albert_graph(N, M)
    G = igraph.Graph(list(G_nx.edges()))
    return G_nx, G

def generateErdos_RenyiNetwork(N=0, p=0):
    G_nx = nx.erdos_renyi_graph(N, p, directed=False)
    G = igraph.Graph(list(G_nx.edges()))
    return G_nx, G