import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch as th
import torch_geometric as pyg
import networkx as nx

from .distances import *

def generatePSNetwork(N:int, avg_k:int, gamma:int, T:int, seed=47, **kwargs):
    """Generates a network based on the Popularity-Similarity model [1]. This model assumes a 2-dimensional hyperbolic space in which the networks grow, giving the network room for two measures of node characteristics: \\
                1) Popularity, which affects hierarchy in the network, indicated by radius. \\
                2) Similarity, which affects arbitrary "likeness" between nodes, indicated by their angular distance.

        [1] Papadopoulos, F., Kitsak, M., Ángeles Serrano, M., Boguñá, M., & Krioukov, D. (2012). Popularity versus similarity in growing networks. Nature, 489(7417), 537–540. https://doi.org/10.1038/nature11459 

    Args:
        N (int): Number of nodes in network
        avg_k (even int): Average node degree. Must be even. If odd, will be rounded down to even.
        gamma (float): Scale factor, which controls popularity fading in network growth (as in scale-free networks, typical range is [2,3])
        T (float): Temperature, which controls the probability of random links being generated across larger distances (i.e. affects clustering)
        seed (int, optional): Seed for randomness (set by PyTorch's manual_seed method). Defaults to 0.
        **kwargs:
                |
                |--> 'normalize_radius' (bool). Whether to normalize maximum network radius to 1. Defaults to False.


    Returns:
        pyg.data.Data: A PyTorch-Geometric Data object, containing the newly created network. Saves the network generation attributes (N, avg_k, etc.).
    """

    th.manual_seed(seed)

    data = pyg.data.Data(is_directed=False)
    # Set data attributes indicating it is a network generated by PS model.
    data.is_PS = True
    data.num_nodes = N
    data.avg_k = avg_k
    data.gamma = gamma
    data.T = T
    data.seed = seed
    # Initialize node positions as 2D torch tensor (r, theta). Both are zero
    data.node_polar_positions = th.zeros(size=(data.num_nodes,2))
    # Set node angular positions to random uniform distribution [0, 2π). We
    # can do this now because it doesn't affect PS algorithm.
    data.node_polar_positions[:,1] = th.rand(size=(data.num_nodes,))*2*th.pi
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
        data.node_polar_positions[:t,0] = 2*(1-beta)*r_t + 2*beta*r_s
        data.node_polar_positions[t,0]  = 2*r_t

        #R_t is the radius of the circle containing the network in Euclidean space,
        #which in turn contains the entirety of the Hyperbolic space
        R_t = 0
        # If beta == 1, popularity fading is not simulated
        if beta == 1: R_t = 2*r_t
        elif beta < 1 and T == 0: R_t = 2*r_t - 2*np.log((2*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.pi*m*(1 - beta))) #[1], Eq. 10
        else: R_t = 2*r_t - 2*np.log((2*T*(1 - np.exp(-0.5*(1 - beta)*2*r_t)))/(np.sin(T*np.pi)*m*(1 - beta)))         #[1], Eq. 26
        
        #save all hyperbolic distances between new node and other nodes, and sort from shortes to longest
        #this sort method returns a tensor with two sub-tensors: d.values and d.indices (with indices sorted by values)
        d = hyperbolic_distances(data.node_polar_positions[:t-1], data.node_polar_positions[t]).sort()
        
        #If T = 0, simply connect to the m hyperbolically closest nodes
        if T == 0:
            new_edges = th.stack([d.indices[:m], th.empty(m).fill_(t)])
            data.edge_index = th.cat((data.edge_index, new_edges), dim=1)
            data.edge_index = pyg.utils.to_undirected(data.edge_index)
        
        else:
            # probability that the new node connects to the other nodes in the network
            p = 1 / (1 + th.exp((d.values - R_t)/(2*T)))
            p = th.nan_to_num(p, nan=0)
            # get m nodes to connect to, sampled by the probabilities given by p.values
            selected_nodes = np.random.choice(d.indices.detach().numpy(), size=m, p=(p/th.sum(p)).detach().numpy(), replace=False)
            new_edges = th.stack([th.Tensor(selected_nodes), th.empty(m).fill_(t)])
            data.edge_index = th.cat((data.edge_index, new_edges), dim=1)
            data.edge_index = pyg.utils.to_undirected(data.edge_index)
    
    data.edge_index = data.edge_index.type(th.int64)
    if kwargs.get('normalize_radius', False):
        # normalize radius to 1
        data.node_polar_positions[:,0] /= data.node_polar_positions[:,0].max()

    return data

def drawPSNetwork(PS:pyg.data.Data, **kwargs):
    """ Function to plot a PS like network. More generally, it can accept a PyTorch-Geometric Data object with coordinates attached. The coordinates are assumed to be in the form of a 2D torch tensor, which can be either cartesian or polar coordinates. If no specific coordinate names are passed (such as those given by an embedding), the network is assumed to have 'node_polar_positions' as attribute, which is given when PS network are generated by `generatePSNetwork`.

    Args:
        PS (pyg.data.Data): The network to be plotted. Must have coordinates attached.
        **kwargs: Additional keyword arguments to be passed to pyg.plot.draw_graph.
                |
                |--> 'figsize' (tuple): The size of the figure to be plotted. \\
                |--> 'polar_projection' (bool): Whether to plot in polar coordinates. Defaults to False. \\
                |--> 'node_size' (int): Size of nodes. Defaults to 50. \\
                |--> 'with_labels' (bool): Whether to label nodes. Defaults to False. \\
                |--> 'pos_name' (str): Name of the position attribute. Defaults to 'node_polar_positions'.

    Returns:
        (plt.figure, plt.ax): The matplotlib figure and axis on which the network is plotted.
    """
    
    if kwargs.get('polar_projection'): subplot_kw={'projection': 'polar'}
    else: subplot_kw=None

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10,10)), subplot_kw=subplot_kw)
    PS_nx = pyg.utils.to_networkx(PS, to_undirected=True)
    
    # control node size by degree
    #degrees = pyg.utils.degree(PS.edge_index[0]).detach().numpy()
    #max_d = np.max(degrees)
    #min_d = np.min(degrees)
    #node_size = 2*((1000/(1+np.exp( -((degrees-min_d)/(max_d-min_d)-0.6)*0.5 ))) + 10).astype(np.int64)
    
    named_positions = getattr(PS, kwargs.get('pos_name', 'node_polar_positions')).detach().numpy()
    pos = dict(zip(range(PS.num_nodes), np.flip(named_positions, axis=1)))

    if hasattr(PS, 'node_polar_positions'): node_color = PS.node_polar_positions[:,1].detach().numpy()
    else: node_color = 'cornflowerblue'

    nx.draw(PS_nx, ax=ax, pos=pos, node_color=node_color, cmap=plt.cm.rainbow,
                          node_size=kwargs.get('node_size', 50), width=0.2, with_labels=kwargs.get('with_labels', False))

    return fig, ax