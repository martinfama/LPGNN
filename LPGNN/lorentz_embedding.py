from importlib.metadata import requires
from tqdm import tqdm
import numpy as np
import torch as th
import torch_geometric as pyg

from .distances import to_cartesian, to_spherical
from .labne import radial_ordering

from .embedding import *


class Model(th.nn.Module):
    """ Incorporates the lorentz embedding into a PyTorch model. """
    def __init__(self, dim, size, init_weights=1e-3, init_pos=None, epsilon=1e-7, R=1):
        super().__init__()
        self.embedding = th.nn.Embedding(size, dim, sparse=False)
        if init_pos is not None:
            self.embedding.weight.data = init_pos
        else:
            self.embedding.weight.data.uniform_(-init_weights, init_weights)
            self.embedding.weight.data[:,0] = th.sqrt(1+th.norm(self.embedding.weight.data[:,1:], dim=1))
        self.epsilon = epsilon
        self.R = R

    def lorentz_metric(self, p: th.Tensor, u: th.Tensor):
        return -p[:,:,0]*u[:,:,0] + th.sum(p[:,:,1:]*u[:,:,1:], dim=-1)

    def dist(self, u, v):
        """ Compute the lorentz distance between two vectors. """
        # ensure that vectors are on hyperboloid
        
        d = th.arccosh( -self.lorentz_metric(u,v) )
        return d

    def forward(self, inputs):
        e = self.embedding(inputs)
        o = e.narrow(dim=1, start=1, length=e.size(1) - 1)
        s = e.narrow(dim=1, start=0, length=1).expand_as(o)

        return self.dist(s, o)

# approximates the exponential map (the retraction)
# this is what the authors in [1] use
@th.jit.script
def approx_expm(p: th.Tensor, u: th.Tensor):
    return p + u

def lorentz_metric(p: th.Tensor, u: th.Tensor):
    return -p[:,0]*u[:,0] + th.sum(p[:,1:]*u[:,1:], dim=-1)

# exact exponential map
@th.jit.script
def exact_expm(p: th.Tensor, u: th.Tensor):
    u_sqrd = th.clamp(th.sqrt(lorentz_metric(u,u)), min=1e-09)
    th.nan_to_num_(u_sqrd, nan=1e-09)
    expm = th.cosh(u_sqrd) * p + th.sinh(u_sqrd) * u / u_sqrd
    return expm

# the Riemannian gradient on the Poincaré ball 
@th.jit.script
def grad(p: th.Tensor):
    g_l = th.ones(p.shape[1])
    g_l[0] = -1
    return p.grad.data * g_l.expand_as(p.grad.data)

# Riemannian stochastic gradient descent
class RiemannianSGD(th.optim.Optimizer):
    def __init__(self, params, expm='approx'):
        super(RiemannianSGD, self).__init__(params, {})
        if expm == 'approx':
            self.expm = approx_expm
        elif expm == 'exact':
            self.expm = exact_expm

    def step(self, lr=0.3, expm='approx'):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = grad(p)
                d_p.mul_(-lr)

                p.data.copy_(self.expm(p.data, d_p))
                # set p.data[0] 
                p.data[:,0] = th.sqrt(1+th.norm(p[:,1:], dim=1)**2)
                print(lorentz_metric(p.data, p.data))
                #return lorentz_metric(p.data, p.data)

def lorentz_embedding(data:pyg.data.Data, edge_index_name='edge_index', epochs=100, lr = 0.03, init_pos=None, dim=2, expm='approx', normalize_radius=False, R=1):
    """ Embed a graph using the lorentz embedding."""

    th.set_default_dtype(th.float64)
    # get the edge index which we want to use
    edge_index = getattr(data, edge_index_name)
    # degrees = pyg.utils.degree(edge_index[0], dtype=th.float64)
    # degrees = degrees / degrees.sum()
    # cat_dist = th.distributions.Categorical(degrees)
    # unif_dist = th.distributions.Categorical(probs=th.ones(data.num_nodes,) / data.num_nodes)
    
    # create the Poincaré Embedding model
    model = Model(size=data.num_nodes, init_pos=init_pos, dim=dim, R=R)
    # for gradient optimizer, we use the Riemannian SGD
    optimizer = RiemannianSGD(model.parameters(), expm=expm)

    # a simple CrossEntropyLoss is used
    loss_func = th.nn.CrossEntropyLoss()

    # pyg.loader.NeighborSampler is used to sample neighbors of nodes
    neighbor_sampler = pyg.loader.NeighborSampler(edge_index, sizes=[-1])
    
    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        
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
            optimizer.step(lr=lr)
        except:
            pass

    data_lorentz = data.clone()
    if normalize_radius:
        data_lorentz.LorentzEmbedding_node_positions = model.embedding.weight / model.embedding.weight.norm(dim=1, keepdim=True) * normalize_radius
    else:
        data_lorentz.LorentzEmbedding_node_positions = model.embedding.weight
    data_lorentz.LorentzEmbedding_node_polar_positions = to_spherical(data_lorentz.LorentzEmbedding_node_positions)

    return data_lorentz