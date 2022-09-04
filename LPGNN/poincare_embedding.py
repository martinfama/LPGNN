from tqdm import tqdm
import numpy as np
import torch as th
import torch_geometric as pyg
from gensim.models.poincare import PoincareModel

from .embedding import *

class Model(th.nn.Module):
    def __init__(self, dim, size, init_weights=1e-3, init_pos=None, epsilon=1e-7):
        super().__init__()
        self.embedding = th.nn.Embedding(size, dim, sparse=False)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        if init_pos is not None:
            self.embedding.weight.data = init_pos
        self.epsilon = epsilon

    def dist(self, u, v):
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

@th.jit.script
def expm(p: th.Tensor, u: th.Tensor):
    return p + u
    # for exact exponential mapping
    #norm = th.sqrt(th.sum(u ** 2, dim=-1, keepdim=True))
    #return mobius_add(p, th.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))

@th.jit.script
def grad(p: th.Tensor):
    p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)

class RiemannianSGD(th.optim.Optimizer):
    def __init__(self, params):
        super(RiemannianSGD, self).__init__(params, {})

    def step(self, lr=0.3):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p.mul_(-lr)

                p.data.copy_(expm(p.data, d_p))

def poincare_embedding(data:pyg.data.Data, edge_index='edge_index', DIMENSIONS=2, epochs=100, init_pos=None):

    th.set_default_dtype(th.float64)
    edge_index = getattr(data, edge_index)
    degrees = pyg.utils.degree(edge_index[0], dtype=th.float64)
    degrees = degrees / degrees.sum()
    # cat_dist = th.distributions.Categorical(degrees)
    # unif_dist = th.distributions.Categorical(probs=th.ones(data.num_nodes,) / data.num_nodes)
    model = Model(dim=DIMENSIONS, size=data.num_nodes, init_pos=init_pos)
    optimizer = RiemannianSGD(model.parameters())

    loss_func = th.nn.CrossEntropyLoss()

    lr = 0.3
    neighbor_sampler = pyg.loader.NeighborSampler(edge_index, sizes=[-1])
    epoch = 0
    while True:
        if epoch == epochs:
            break
        if epoch < 0:
            lr = 0.003
        else:
            lr = 0.03
        epoch += 1
        
        #for random_node in np.arange(data.num_nodes):
        random_node = np.random.randint(0, data.num_nodes)
        node_neighbors_e_id = neighbor_sampler.sample([random_node])[2][1]
        pos_edges = edge_index[:, node_neighbors_e_id]
        batch_X = th.zeros(pos_edges.shape[1], 3, dtype=th.long)
        batch_y = th.zeros(pos_edges.shape[1], dtype=th.long)
        batch_X[:,:2] = th.fliplr(pos_edges.T)
        
        neg_nodes = th.arange(data.num_nodes).float()
        weight = th.ones_like(neg_nodes)
        weight[pos_edges[0]] = 0
        try:
            neg_nodes = neg_nodes[th.multinomial(weight, pos_edges.shape[1], replacement=False)].long()
            batch_X[:,2] = neg_nodes.T
            
            optimizer.zero_grad()
            preds = model(batch_X)

            loss = loss_func(preds.neg(), batch_y)
            loss.backward()
            optimizer.step(lr=lr)
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

