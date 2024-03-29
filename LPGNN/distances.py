#!/usr/bin/env python

"""Provides a suite of functions related to calculating distances in Euclidean and Hyperbolic space.
Includes functions which simply calculate the distances between position lists, and also functions
to save to file, read from file, etc. All inputs and outputs are given as pytorch Tensors.
"""

from typing import Optional
import pandas as pd
import torch as th
from .Logger import print

def to_spherical(u:th.Tensor):
    """ Assumes a N-dimensional vector in cartesian coordinates (x_1, x_2, ..., x_n).
    Returns a N-dimensional vector in spherical coordinates (r, theta_1, theta_2, ..., theta_n-1). """

    reshape = False
    if u.dim() == 1:
        # reshape u
        u = u.reshape(1,-1)
        reshape = True
    u_s = th.zeros_like(u)
    u_s[:,0] = th.norm(u, dim=1)
    for i in range(1, u_s.shape[1]-1):
        # set hyperspherical coordinates
        u_s[:,i] = th.acos(u[:,i-1] / th.norm(u[:,i-1:], dim=1))
    # set polar coordinate
    u_s[:,-1] = th.arccos(u[:,-2] / th.sqrt(u[:,-1]**2 + u[:,-2]**2))
    u_s[:,-1][u[:,-1] < 0] = 2*th.pi - u_s[:,-1][u[:,-1] < 0]
    u_s = th.nan_to_num_(u_s, nan=0)
    if reshape: u_s = u_s.reshape(-1)
    return u_s

def to_cartesian(u:th.Tensor):
    """ Assumes a N-dimensional vector in spherical coordinates (r, theta_1, theta_2, ..., theta_n-1).
    Returns a N-dimensional vector in cartesian coordinates (x_1, x_2, ..., x_n). """

    reshape = False
    if u.dim() == 1:
        # reshape u
        u = u.reshape(1,-1)
        reshape = True
    u_c = th.zeros_like(u)
    for i in range(u_c.shape[1]-1):
        # set cartesian coordinates
        u_c[:,i] = u[:,0] * th.prod(th.sin(u[:,1:i+1]), dim=1) * th.cos(u[:,i+1])
    u_c[:,-1] = u[:,0] * th.prod(th.sin(u[:,1:]), dim=1)
    if reshape: u_c = u_c.reshape(-1)
    return u_c

def hyperbolic_distance(u:th.Tensor, v:th.Tensor):
    """ Calculates hyperbolic distances between positions given as a list of [r, theta] values.
        Uses idx to indicate which positions to compare.

    Args:
        positions (th.Tensor): A list of [r,theta] values
        idx (th.Tensor): A tensor of shape [N,2] indicating the combination of positions to compare.

    Returns:
        th.Tensor: A 1D tensor of hyperbolic distances of length N.
    """

    # ensure u,v type is double
    u = u.double()
    v = v.double()

    # get the angular distance by applying acos() to the dot product of the two vectors
    r_u = th.norm(u, dim=1)
    r_v = th.norm(v, dim=-1)
    dot = th.sum(u * v, dim=1)
    angular_distance = th.acos(dot / (r_u * r_v))
    #angular_distance = th.min(2*th.pi-th.abs(u[:,1]-v[:,1]), th.abs(u[:,1]-v[:,1]))

    d = th.arccosh(th.cosh(r_u)*th.cosh(r_v) - th.sinh(r_u)*th.sinh(r_v)*th.cos(angular_distance))
    return d

def hyperbolic_distance_from_spherical_2D(u:th.Tensor, v:th.Tensor):
    """ Calculates hyperbolic distances between positions given as a list of [r, theta] values.
        Uses idx to indicate which positions to compare.

    Args:
        positions (th.Tensor): A list of [r,theta] values
        idx (th.Tensor): A tensor of shape [N,2] indicating the combination of positions to compare.

    Returns:
        th.Tensor: A 1D tensor of hyperbolic distances of length N.
    """

    # ensure u,v type is double
    u = u.double()
    v = v.double()

    # since u and v are of the form [r, theta], we can use the following formula to calculate the
    # hyperbolic distance between them
    angular_distance = th.min(2*th.pi-th.abs(u[:,1]-v[:,1]), th.abs(u[:,1]-v[:,1]))
    r_u = u[:,0]
    r_v = v[:,0]

    d = th.arccosh(th.cosh(r_u)*th.cosh(r_v) - th.sinh(r_u)*th.sinh(r_v)*th.cos(angular_distance))
    return d

def poincare_distance(u:th.Tensor, v:th.Tensor, max_r=1):
    """ Compute the Poincare distance between two vectors. """

    # ensure u,v type is double
    u = u.double()
    v = v.double()

    sqdist = th.sum((u - v) ** 2, dim=-1) 
    squnorm = th.sum(u ** 2, dim=-1)
    sqvnorm = th.sum(v ** 2, dim=-1) 
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-15
    z = th.sqrt(x ** 2 - 1)
    return th.log(x + z)
    # sqdist = max_r**2 * th.sum((u - v) ** 2, dim=-1)
    # squnorm = max_r**2 - th.sum(u ** 2, dim=-1)
    # sqvnorm = max_r**2 - th.sum(v ** 2, dim=-1)
    
    # return th.arccosh(1 + 2 * sqdist / (squnorm * sqvnorm))

def gen_dist_list(positions:th.Tensor, dist='hyp'):
    """ Calculates _all_ hyperbolic distances by comparing all positions. Since the memory overhead
        can become huge for moderately large graphs (more than 10.000 nodes), we save these values to
        a file in chunks. The save format is:
            idx_1, idx_2, distance
            (int), (int), (float)
             ... ,  ... ,   ...
        where idx_1 and idx_2 are the indices of the positions compared, and distance is the hyperbolic distance.

    Args:
        positions (th.Tensor): The list of positions given as [r,theta] values
        chunk_size (int): What length lists we should save in.
        filename (str): File to save to.
        extra_info_tensor (th.Tensor, optional): A tensor of extra information to save to file. Defaults to None. Implemented to save the edge labels (i.e. 0 or 1).
        skip_index (th.Tensor, optional): A tensor of indices to skip. Defaults to None.
        dist (str, optional): The distance metric to use. Defaults to 'poincare'. Options are 'poincare' and 'hyp'.
    """

    if dist == 'poincare': dist_func = poincare_distance
    elif dist == 'hyp': dist_func = hyperbolic_distance
    elif dist == 'hyp_spherical': dist_func = hyperbolic_distance_from_spherical_2D

    N = positions.shape[0]
    # change positions type to th.DoubleTensor
    positions = positions.double()
    
    idx = th.triu_indices(*(N, N), offset=1)
    d = dist_func(positions[idx[0]], positions[idx[1]])
    return d, idx

def hyperbolic_distance_list_to_file(positions:th.Tensor, chunk_size:int, filename:str, extra_info_tensor:Optional[th.Tensor] = None, skip_index:Optional[th.Tensor] = None, dist='poincare'):
    """ Calculates _all_ hyperbolic distances by comparing all positions. Since the memory overhead
        can become huge for moderately large graphs (more than 10.000 nodes), we save these values to
        a file in chunks. The save format is:
            idx_1, idx_2, distance
            (int), (int), (float)
             ... ,  ... ,   ...
        where idx_1 and idx_2 are the indices of the positions compared, and distance is the hyperbolic distance.

    Args:
        positions (th.Tensor): The list of positions given as [r,theta] values
        chunk_size (int): What length lists we should save in.
        filename (str): File to save to.
        extra_info_tensor (th.Tensor, optional): A tensor of extra information to save to file. Defaults to None. Implemented to save the edge labels (i.e. 0 or 1).
        skip_index (th.Tensor, optional): A tensor of indices to skip. Defaults to None.
        dist (str, optional): The distance metric to use. Defaults to 'poincare'. Options are 'poincare' and 'hyp'.
    """

    if dist == 'poincare': dist_func = poincare_distance
    elif dist == 'hyp': dist_func = hyperbolic_distance
    elif dist == 'hyp_spherical': dist_func = hyperbolic_distance_from_spherical_2D

    N = positions.shape[0]
    # change positions type to th.DoubleTensor
    positions = positions.double()
    # Get the indices of node pairs corresponding to the upper triangle of the distance matrix,
    # since the distance matrix is symmetric. We omit the diagonal by setting offset to 1.
    idx = th.triu_indices(*(N, N), offset=1)
    
    # auxiliary function to save a chunk of distances to file. Takes as input the indices of the
    # positions to compare. 
    def _(rng):
        idx_t = idx[:,rng]
        d = dist_func(positions[idx_t[0]], positions[idx_t[1]])
        d = th.nan_to_num(d, nan=th.inf)
        if extra_info_tensor is None: 
            d = pd.DataFrame(th.stack([*idx_t, d], dim=0).T.detach().numpy())
            d[[2]] = d[[2]].astype(float).round(21)
        else: 
            d = pd.DataFrame(th.stack([*idx_t, extra_info_tensor[rng], d], dim=0).T.detach().numpy())
            d[[2]] = d[[2]].astype(int)
            d[[3]] = d[[3]].astype(float).round(21)

        d[[0,1]] = d[[0,1]].astype(int)
        d.to_csv(filename, mode='a', header=False, index=False)
    
    # iterate over the indices in chunks of chunk_size.
    for index in range(0, idx.shape[1]-chunk_size, chunk_size):
        rng = range(index, index+chunk_size)
        _(rng)
    # get what's leftover of idx.
    rng = range(index+chunk_size, idx.shape[1])
    _(rng)