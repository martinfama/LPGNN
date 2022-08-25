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
from .utils import infer_gamma

#generates the Laplacian Based Network Embedding for a given Network
def generateLaBNE(data:pyg.data.Data, only_coordinates=False):
    """ Given a graph, returns the LaBNE embedding as described in [1].

        [1] Alanis-Lobato, G., Mier, P. & Andrade-Navarro, M. Efficient embedding of complex networks to hyperbolic space via their Laplacian. Sci Rep 6, 30108 (2016). https://doi.org/10.1038/srep30108

    Args:
        data (pyg.data.Data): The graph to embed.
        only_coordinates (Bool): Whether to return only the coordinates generated by LaBNE, or a new graph object containing these coordinates. Defaults to False.

    Returns:
        if only_coordinates:
            (th.Tensor, th.Tensor, th.Tensor, th.Tensor): The torch Tensor's which represente x,y,r,theta coordinates for each node.
        else:
            pyg.data.Data: A copy of the inputted data object, but with coordinates set to LaBNE.
    """

    N = data.num_nodes

    #get Laplacian matrix of the graph (L = D - A). We pass it to a sparse matrix type supported by SciPy
    #so that we can use scipy's sparse linear algebra tools
    L = pyg.utils.get_laplacian(data.edge_index)
    L = pyg.utils.to_scipy_sparse_matrix(L[0], L[1])
    #the tol parameter of scipy.sparse.lingalg() is to give an error tolerance. for smaller networks, convergence seems to be
    #guaranteed with high precision. for larger networks, convergence maybe a problem, so an error tolerance is given. criteria
    #the same as in Alanis-Lobato's NetHypGeom library
    if N < 10000:
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=3, which='LM', sigma=0, return_eigenvectors=True)
        except(RuntimeError):
            Levenberg_c = np.zeros(np.shape(L))
            np.fill_diagonal(Levenberg_c, 0.01)
            Levenberg_c = scipy.sparse.coo_matrix(Levenberg_c)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L+Levenberg_c, k=3, which='LM', sigma=0, return_eigenvectors=True)
    else:        
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L, k=3, which='LM', sigma=0, return_eigenvectors=True, tol=1E-7, maxiter=5000)
        except(RuntimeError):
            Levenberg_c = np.zeros(np.shape(L))
            np.fill_diagonal(Levenberg_c, 0.01)
            Levenberg_c = scipy.sparse.coo_matrix(Levenberg_c)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A=L+Levenberg_c, k=3, which='LM', sigma=0, return_eigenvectors=True, tol=1E-7, maxiter=5000)

    eigenvectors = np.transpose(eigenvectors)
    x, y = th.Tensor(eigenvectors[1].real), th.Tensor(eigenvectors[2].real)
    r = th.zeros(N)
    theta = th.atan2(y, x)
    theta = (theta + 2*th.pi) % (2*th.pi)

    degrees = pyg.utils.degree(data.edge_index[0])
    # sort degrees for radial positioning
    degrees = degrees.sort(descending=True)
    m = int(degrees.values.mean()/2)
    gamma = infer_gamma(data).power_law.alpha
    beta = 1/(gamma-1)
    for t in range(N):
        r_t = np.log(t+1)
        r[degrees.indices[t]] = 2*beta*r_t + 2*(1-beta)*np.log(N)

    if only_coordinates:
        return x, y, r, theta

    data_LaBNE = data.clone()
    data_LaBNE.LaplacianEigenmaps_node_positions = th.stack((x,y)).T
    data_LaBNE.LaBNE_node_polar_positions = th.stack((r,theta)).T
    return data_LaBNE