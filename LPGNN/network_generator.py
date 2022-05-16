__version__ = '1.2.3'

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
        # Temp is the network temperature, which is related to it's clustering coefficient \bar{c}
        if 'Temp' in kwargs:
            self.Temp = kwargs.get('Temp')
    
    def generatePSNetwork(self, N=None, avg_k=None, gamma=None, Temp=None, seed=None):
        
        if N != None:
            self.N = N
        if avg_k != None:
            self.avg_k = avg_k
        if gamma != None:
            self.gamma = gamma
        if Temp != None:
            self.Temp = Temp
        if seed != None:
            self.seed = time.time()
        
        self.graph = generatePSNetwork(N, avg_k, gamma, Temp, seed)

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
def generatePSNetwork(N=0, avg_k=0, gamma=0, Temp=0, seed=0):

    random.seed(seed)
    
    #beta controls popularity fading
    beta = 1/(gamma-1)
    m = round(avg_k/2)
    
    #create a Graph() object. we add the first node, and set it's attributes (original_index, r and theta)
    graph = igraph.Graph()
    graph.add_vertex()
    #original index
    graph.vs['o_i'] = [0]
    #r coordinate
    graph.vs['r'] = [0]
    #angular coordinate
    graph.vs['theta'] = [random.uniform(0, 2*np.pi)]

    #t acts as our discrete time variable, where an increase of 1 indicates a new node entering the network
    #special attention must be paid to whether t is used as an _index_ for lists, in which case it is used
    #as is, or when it is used related to the network itself, where t+1 must be used, since we start the count
    #at 1
    for t in range(1, N):
        r_t = np.log(t+1)

        #update all of the existing nodes radial coordinates according to
        # r_s(t) = beta*r_s + (1-beta)*r_t
        #which simulates popularity decay due to aging
        for s in range(t):
            r_s = np.log(s+1)
            graph.vs[s]['r'] = 2*(1-beta)*np.log(t+1) + 2*beta*r_s

        #R_t is the radius of the circle containing the network in Euclidean space,
        #which in turn contains the entirety of the Hyperbolic space
        R_t = 0
        # If beta == 1, popularity fading is not simulated
        if beta == 1:
            R_t = 2*r_t
        elif beta < 1 and Temp == 0:
            R_t = 2*r_t - 2*np.log((2*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.pi*m*(1 - beta))) #SI, Eq. 10
        else:
            #R_t = 2*r_t - 2*np.log((2*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.pi*m*(1 - beta))) #SI, Eq. 10
            R_t = 2*r_t - 2*np.log((2*Temp*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.sin(Temp*np.pi)*m*(1 - beta))) #SI, Eq. 26
            
        #add new node and set attributes
        graph.add_vertex()
        graph.vs[-1]['o_i'] = t
        graph.vs[-1]['r'] = 2*np.log(t+1)
        graph.vs[-1]['theta'] = random.uniform(0, 2*np.pi)
        
        #if t <= m, we simply connect to all existing nodes in the network
        if t <= m:
                for s in range(t):
                    graph.add_edge(t, s)
        else:
            #save all hyperbolic distances between new node and other nodes 
            s_unconnected = [s for s in range(t) if (not graph.are_connected(t, s))]
            hyperbolic_distances = np.array([[s, hyperbolic_distance(graph.vs[t]['r'], graph.vs[t]['theta'], graph.vs[s]['r'], graph.vs[s]['theta'])] for s in s_unconnected])

            if Temp > 0:
                #probability that the new node connects to the other nodes in the network
                d = np.transpose(hyperbolic_distances)[1]
                p = 1 / (1 + np.exp((d - R_t)/(2*Temp)))
                #get the subset of nodes that are not connected to t

                m_count = 0
                while m_count < m:
                    #print(f'\r{m_count}', end='')
                    #this is to check if the probability is relatively high of having connections
                    #if it is not, we simply connect to the m closest nodes
                    if ( sum(p[s] < 0.1 for s in range(len(s_unconnected))) == len(s_unconnected) ):
                        #order hyperbolic distances
                        hyperbolic_distances = sorted(hyperbolic_distances, key=lambda d: d[1])
                        #connect to m closest nodes
                        while m_count < m:
                            graph.add_edge(t, int(hyperbolic_distances[m_count][0]))
                            m_count += 1
                    #otherwise, we connect to m nodes randomly, with their respective probabilities
                    else:
                        #sample nodes randomly, and connect to them with probability p, until m nodes are reached. keep a list of unconnected nodes
                        s_try = np.random.choice(s_unconnected, size=m, p=p/sum(p))
                        #if random.uniform(0,1) <= p[s_try]:
                        for s in s_try:
                            if not graph.are_connected(t, s):
                                graph.add_edge(t, s)
                                index = np.where(s_unconnected == s)
                                s_unconnected = np.delete(s_unconnected, index)
                                hyperbolic_distances = np.delete(hyperbolic_distances, (np.where(np.transpose(hyperbolic_distances)[:,0]==s)), axis=0)
                                p = np.delete(p, index)
                                #hyperbolic_distances = hyperbolic_distances[:,0] != s_try
                                m_count += 1
            
            #If Temp = 0, simply connect to the m hyperbolically closest nodes
            else:
                #order hyperbolic distances
                hyperbolic_distances = sorted(hyperbolic_distances, key=lambda d: d[1])
                #the first element of each row of hyperbolic_distances is the node index and we ordered hyperbolic_distances, so the first m entries are the m closest nodes
                for i in range(m):
                    if (graph.are_connected(t, int(hyperbolic_distances[i][0]))):
                        print(f"Temp 0, {t}, {hyperbolic_distances[i][0]}")
                    graph.add_edge(t, int(hyperbolic_distances[i][0]))

    #give the igraph Graph x and y coordinates, so as to be able to plot it
    #and show the hyperbolic structure
    graph.vs['x'] = graph.vs['r']*np.cos(graph.vs['theta'])
    graph.vs['y'] = graph.vs['r']*np.sin(graph.vs['theta'])
    graph.vs['degree'] = graph.degree()
    #set the 'size' attribute of each node. nodes with a higher degree are bigger
    graph.vs['size'] = node_size_by_degree(graph=graph)

    #assign a rainbow palette based on the angular coordinate
    rainbow_palette = igraph.RainbowPalette(n=N)
    for i in range(N):
        graph.vs[i]['color'] = rainbow_palette.get(int(graph.vs[i]['theta']/(2*np.pi)*N))
    
    return graph

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


    #Temp = network.Temp
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
            labne_graph.vs[i]['color'] = rainbow_palette.get(int(labne_graph.vs[i]['theta']/(2*np.pi)*N))

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

def generatePSNetwork_nx(N=0, avg_k=0, gamma=0, Temp=0, seed=0):
    PS = generatePSNetwork(N=N, avg_k=avg_k, gamma=gamma, Temp=Temp, seed=seed)
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