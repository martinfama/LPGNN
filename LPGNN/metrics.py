from math import dist
import igraph
import numpy as np
import torch as th

#import network_generator as ng
#import network_analysis as na

def hyperbolic_distance(r_1, theta_1, r_2, theta_2):
    
    #get angular distance between nodes
    #we need the net angular distance, not simply the difference between angles
    #for example, a node at 0.2 and another at 2*Pi-0.2 have an angular distance of
    #0.4, not 2*Pi-0.4
    angular_distance = min(2*np.pi-np.abs(theta_1-theta_2), np.abs(theta_1-theta_2))
    
    #return the hyperbolic distance between the two nodes
    d = np.arccosh(np.cosh(r_1)*np.cosh(r_2) - np.sinh(r_1)*np.sinh(r_2)*np.cos(angular_distance))
    if np.isnan(d):
        return 0
    else:
        return d

def hyperbolic_distances(position_list:th.Tensor, position_single:th.Tensor):
    """Gets the hyperbolic distances between ``position_single`` and list ``position_list``

    Args:
        position_list (th.Tensor): A (N,2) tensor of floats (which indicate (r, theta))
        position_single (th.Tensor): A 2 element tensor which indicates specific (r, theta) pos

    Returns:
        th.Tensor: Tensor of distances, shape (N,1)
    """
    
    #get angular distance between list of positions and index position
    angular_distance = th.min(2*th.pi-th.abs(position_list[:,1]-position_single[1]), th.abs(position_list[:,1]-position_single[1]))
    
    #calculate hyperbolic distance between list of positions and index position
    d = th.arccosh(th.cosh(position_single[0])*th.cosh(position_list[:,0]) - th.sinh(position_single[0])*th.sinh(position_list[:,0])*th.cos(angular_distance))
    return d

def hyperbolic_distance_matrix(positions:th.Tensor):
    N = positions.shape[0]
    d = th.Tensor([])
    f = open('temp.txt', 'a')
    for i in range(N):
        #distances = th.stack((th.full((N-i-1,), i), th.arange(i+1, N), hyperbolic_distances(positions[i+1:], positions[i])))
        distances = hyperbolic_distances(positions[i+1:], positions[i])
        # d = th.cat((d, distances), dim=1)
        d = th.cat((d, distances), dim=0)
        if i % 100 == 0:
            np.savetxt(f, d.detach().numpy())       
            print('\r', i, distances.shape, d.shape, end='')
            d = th.Tensor([])
    return d

# metrics for node pairs. define all in a standard way so that the precision-recall function
# has a consistent interface, and you can select the metrics you want to use by name

## Neighbourhood-based link predictors
#label: 'CN'
def common_neighbors(graph=igraph.Graph(), i=int, j=int, **kwargs):
    return len(set(graph.neighbors(i)).intersection(graph.neighbors(j)))#/np.sqrt(graph.degree(i)*graph.degree(j))
#label: 'DS'
def dice_similarity(graph=igraph.Graph(), i=int, j=int, **kwargs):
    try:
        return 2*len(set(graph.neighbors(i)).intersection(graph.neighbors(j)))/(graph.degree(i)+graph.degree(j))
    except:
        return 0
#label: 'AA'
def adamic_adar_index(graph=igraph.Graph(), i=int, j=int, **kwargs):
    return np.sum(np.log(1+np.array(list(graph.neighbors(i))+list(graph.neighbors(j)))))
#label: 'PA'
def preferential_attachment(graph=igraph.Graph(), i=int, j=int, **kwargs):
    try:
        return graph.degree(i)*graph.degree(j)/(graph.degree(i)+graph.degree(j))
    except:
        return 0
#label: 'JC'
def jaccard_coefficient(graph=igraph.Graph(), i=int, j=int, **kwargs):
    return len(set(graph.neighbors(i)).intersection(graph.neighbors(j)))/len(set(graph.neighbors(i)).union(graph.neighbors(j)))

## LaBNE link predictor. We return the Euclidean distance squared between nodes i and j. we assume that the graph that
# is being passed is an embedding with LaBNE.
def LaBNE_metric(graph=igraph.Graph(), i=int, j=int, **kwargs):
    #d = (graph.vs[i]['x'] - graph.vs[j]['x'])**2 + (graph.vs[i]['y'] - graph.vs[j]['y'])**2
    d = hyperbolic_distance(graph.vs[i]['r'], graph.vs[i]['theta'], graph.vs[j]['r'], graph.vs[j]['theta'])
    return d