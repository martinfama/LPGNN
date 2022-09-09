from tqdm import tqdm
import numpy as np
import torch as th
import torch_geometric as pyg
from gensim.models.poincare import PoincareModel

from .embedding import *

class Model(th.nn.Module):
    """ Incorporates the Poincare embedding into a PyTorch model. """
    def __init__(self, dim, size, init_weights=1e-3, init_pos=None, epsilon=1e-7):
        super().__init__()
        self.embedding = th.nn.Embedding(size, dim, sparse=False)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        if init_pos is not None:
            self.embedding.weight.data = init_pos
        self.epsilon = epsilon

    def dist(self, u, v):
        """ Compute the Poincare distance between two vectors. """
        sqdist = th.sum((u - v) ** 2, dim=-1)
        squnorm = th.sum(u ** 2, dim=-1)
        sqvnorm = th.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = th.sqrt(x ** 2 - 1)
        return th.log(x + z)

    def forward(self, inputs):
        e = self.embedding(inputs)
        o = e.narrow(dim=1, start=1, length=e.size(1) - 1)
        s = e.narrow(dim=1, start=0, length=1).expand_as(o)

        return self.dist(s, o)

@th.jit.script
def lambda_x(x: th.Tensor):
    return 2 / (1 - th.sum(x ** 2, dim=-1, keepdim=True))

@th.jit.script
def mobius_add(x: th.Tensor, y: th.Tensor):
    x2 = th.sum(x ** 2, dim=-1, keepdim=True)
    y2 = th.sum(y ** 2, dim=-1, keepdim=True)
    xy = th.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2

    return num / denom.clamp_min(1e-15)

# approximates the exponential map (the retraction)
# this is what the authors in [1] use
@th.jit.script
def approx_expm(p: th.Tensor, u: th.Tensor):
    return p + u
    
# for exact exponential mapping
@th.jit.script
def exact_expm(p: th.Tensor, u: th.Tensor):
    norm = th.sqrt(th.sum(u ** 2, dim=-1, keepdim=True))
    return mobius_add(p, th.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))

# the Riemannian gradient on the Poincaré ball 
@th.jit.script
def grad(p: th.Tensor):
    p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)

# Riemannian stochastic gradient descent
class RiemannianSGD(th.optim.Optimizer):
    def __init__(self, params):
        super(RiemannianSGD, self).__init__(params, {})

    def step(self, lr=0.3, expm='approx'):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p.mul_(-lr)
                
                if expm == 'approx':
                    p.data.copy_(approx_expm(p.data, d_p))
                elif expm == 'exact':
                    p.data.copy_(exact_expm(p.data, d_p))

def poincare_embedding(data:pyg.data.Data, edge_index='edge_index', DIMENSIONS=2, epochs=100, lr = 0.03, init_pos=None, expm='approx'):
    """ Embed a graph using the Poincare embedding.

    Args:
        data (pyg.data.Data): Graph to embed.
        edge_index (str, optional): Name of edge_index to use from data, for example can be 'train_pos_edge_label_index'. Defaults to 'edge_index', i.e. full graph.
        DIMENSIONS (int, optional): Dimensions to embed to. Defaults to 2.
        epochs (int, optional): Number of epochs to run. A single epoch get a single node and updates according to neighbors and negative samples. Defaults to 100.
        init_pos (th.Tensor, optional): Initial position of the embedding. Defaults to None, which in turn initializes randomly.
        expm (str, optional): Exponential mapping to use. Defaults to 'approx', which is faster but less accurate. 'exact' is slower but more accurate.

    Returns:
        pyg.data.Data: A cloned version of data with the embedding added, named 'PoincareEmbedding_node_positions' and 'PoincareEmbedding_node_polar_positions'.
    """

    th.set_default_dtype(th.float64)
    # get the edge index which we want to use
    edge_index = getattr(data, edge_index)
    # degrees = pyg.utils.degree(edge_index[0], dtype=th.float64)
    # degrees = degrees / degrees.sum()
    # cat_dist = th.distributions.Categorical(degrees)
    # unif_dist = th.distributions.Categorical(probs=th.ones(data.num_nodes,) / data.num_nodes)
    
    # create the model
    model = Model(dim=DIMENSIONS, size=data.num_nodes, init_pos=init_pos)
    # for optimizer, we use the Riemannian SGD
    optimizer = RiemannianSGD(model.parameters())

    # a simple CrossEntropyLoss is used
    loss_func = th.nn.CrossEntropyLoss()

    # pyg.loader.NeighborSampler is used to sample neighbors of nodes
    neighbor_sampler = pyg.loader.NeighborSampler(edge_index, sizes=[-1])
    
    for epoch in tqdm(range(epochs)):
        
        #for random_node in np.arange(data.num_nodes):
        # get a random node
        random_node = np.random.randint(0, data.num_nodes)
        # get the neighbors of the node
        node_neighbors_e_id = neighbor_sampler.sample([random_node])[2][1]
        pos_edges = edge_index[:, node_neighbors_e_id]
        # batch_X and batch_y are used to store the positive and negative samples
        # by setting batch_X shape to (pos_edges.shape[1], 3), we are implying that
        # for every positive sample, we have 1 negative sample. This is because one of the 
        # columns of batch_X is the node itself, and the other two are the positive and negative samples.
        batch_X = th.zeros(pos_edges.shape[1], 3, dtype=th.long)
        batch_y = th.zeros(pos_edges.shape[1], dtype=th.long)
        # set the first column to be the random node and the second column to be the neighbors
        batch_X[:,:2] = th.fliplr(pos_edges.T)
        
        # to get negative samples, we randomly sample from the uniform distribution of all nodes except the neighbors
        neg_nodes = th.arange(data.num_nodes).float()
        weight = th.ones_like(neg_nodes)
        weight[pos_edges[0]] = 0 #this excludes the neighbors from the negative samples
        
        # TODO using a try except block for now, since sometimes the negative sampling fails
        try:
            # get pos_edges.shape[1] number of negative samples
            neg_nodes = neg_nodes[th.multinomial(weight, pos_edges.shape[1], replacement=False)].long()
            # set the third column to be the negative samples
            batch_X[:,2] = neg_nodes.T
            
            # apply update step to the model
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = loss_func(preds.neg(), batch_y)
            loss.backward()
            optimizer.step(lr=lr, expm=expm)
        except:
            pass

    data_Poincare = data.clone()
    data_Poincare.PoincareEmbedding_node_positions = model.embedding.weight
    x, y = data_Poincare.PoincareEmbedding_node_positions[:,0], data_Poincare.PoincareEmbedding_node_positions[:,1]
    data_Poincare.PoincareEmbedding_node_polar_positions = th.stack([th.sqrt(x**2 + y**2), th.atan2(y, x)], dim=1)
    return data_Poincare

# @embedding
# def poincare_embedding(data:pyg.data.Data, initial_coordinates=None, epochs=100, only_coordinates=False, **kwargs):
#     """ Apply the Poincaré embedding [1] to the given graph.

#         [1] Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. arXiv. https://doi.org/10.48550/ARXIV.1705.08039 

#     Args:
#         data (pyg.data.Data): The graph to embed.
#     Returns:
#         pyg.data.Data: The graph with the embedding.
#     """
    
#     model = PoincareModel(data.edge_index.T.detach().numpy(), **kwargs)
#     if initial_coordinates is not None:
#         pos = getattr(data, initial_coordinates).detach().numpy()
#         for node in range(data.num_nodes):
#             model.kv.vectors[node] = pos[node]
#     model.train(epochs)

#     embeddings = np.array([model.kv[node] for node in range(data.num_nodes)])
    
#     if only_coordinates:
#         return embeddings
    
#     data_Poincare = data.clone()
#     data_Poincare.PoincareEmbedding_node_positions = th.Tensor(embeddings)
#     return data_Poincare

