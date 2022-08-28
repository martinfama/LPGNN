#!/usr/bin/env python

"""Provides a suite of functions related to calculating distances in Euclidean and Hyperbolic space.
Includes functions which simply calculate the distances between position lists, and also functions
to save to file, read from file, etc. All inputs and outputs are given as pytorch Tensors.
""" 

import pandas as pd
import torch as th

def hyperbolic_distance(positions:th.Tensor, idx:th.Tensor):
    """ Calculates hyperbolic distances between positions given as a list of [r, theta] values.
        Uses idx to indicate which positions to compare.

    Args:
        positions (th.Tensor): A list of [r,theta] values
        idx (th.Tensor): A tensor of shape [N,2] indicating the combination of positions to compare.

    Returns:
        th.Tensor: A 1D tensor of hyperbolic distances of length N.
    """
    angular_distance = th.min(2*th.pi-th.abs(positions[idx[0]][:,1]-positions[idx[1]][:,1]), th.abs(positions[idx[0]][:,1]-positions[idx[1]][:,1]))
    d = th.arccosh(th.cosh(positions[idx[0]][:,0])*th.cosh(positions[idx[1]][:,0]) - th.sinh(positions[idx[0]][:,0])*th.sinh(positions[idx[1]][:,0])*th.cos(angular_distance))
    return d

def hyperbolic_distance_list_to_file(positions:th.Tensor, chunk_size:int, filename:str):
    """ Calculates _all_ hyperbolic distances by comparing all positions. Since the memory overhead
        can become huge for moderately large graphs (more than 10.000 nodes), we save these values to
        a file in chunks. 

    Args:
        positions (th.Tensor): The list of positions given as [r,theta] values
        chunk_size (int): What length lists we should save in.
        filename (str): File to save to.
    """
    N = positions.shape[0]
    #f = open(filename, 'a')
    idx = th.triu_indices(*(N, N), offset=1)
    for index in range(0, idx.shape[1]-chunk_size, chunk_size):
        idx_t = idx[:,index:index+chunk_size]
        d = hyperbolic_distance(positions, idx_t)
        pd.DataFrame(d.detach().numpy()).to_csv(filename, mode='a', header=False, index=False)
    # get what's leftover of idx.
    idx_t = idx[:,index+chunk_size:]
    d = hyperbolic_distance(positions, idx_t)
    pd.DataFrame(d.detach().numpy()).to_csv(filename, mode='a', header=False, index=False)