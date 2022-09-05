import igraph
import numpy as np
import torch as th
import torch_geometric as pyg

# metrics for node pairs. define all in a standard way so that the precision-recall function
# has a consistent interface, and you can select the metrics you want to use by name

def CN(data:pyg.data.Data, u:int, v:int, **kwargs):
    """ Get number of common neighbors between nodes u and v in data.

    Args:
        data (pyg.data.Data): Graph to analyze.
        u (int): Node u
        v (int): Node v

    Returns:
        int: N° of common neighbors between u and v.
    """
    neighbor_sampler = pyg.loader.NeighborSampler(data.edge_index, sizes=[-1])
    n_u = neighbor_sampler.sample([u])[1][1:]
    n_v = neighbor_sampler.sample([v])[1][1:]
    all_n, counts = th.cat((n_u, n_v), dim=0).unique(return_counts=True)
    return all_n[th.where(counts == 2)].shape[0]

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