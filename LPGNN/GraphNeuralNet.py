import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch.nn import init
import pdb

import LPGNN.network_analysis as na
import LPGNN.LinkPrediction as LinkPrediction
import LPGNN.graph_metrics as graph_metrics

device = 'cpu'

sigmoid = nn.Sigmoid()

def hyperbolic_distance(u, v):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 0.001
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)

def hyperbolic_logits(z, pos_edge_index, neg_edge_index):
    d = hyperbolic_distance(u=z[pos_edge_index[0]], v=z[pos_edge_index[1]])
    d = torch.cat((d, hyperbolic_distance(u=z[neg_edge_index[0]], v=z[neg_edge_index[1]])))
    logits = 1/d
    #d, indices = torch.sort(d, descending=False)
    #logits = torch.ones(d.size(0), dtype=torch.float, device=device, requires_grad=True)*10
    #logits[indices[neg_edge_index.size(1):]] *= (-1)
    return logits

def poincare_loss(z, pos_edge_index, neg_edge_index):
    L = torch.Tensor([0])
    for i in range(pos_edge_index.size(1)):
        d_h_pos = hyperbolic_distance(z[pos_edge_index[0][i]], z[pos_edge_index[1][i]])
        neg_edges = [t for t in neg_edge_index.T if t[0] == pos_edge_index[0][i]]
        denominator = torch.Tensor([0])
        for neg_edge in neg_edges:
            d_h_neg = hyperbolic_distance(z[pos_edge_index[0][i]], z[neg_edge[1]])
            denominator += torch.exp(-d_h_neg)
        if not torch.is_nonzero(denominator):
            denominator = torch.Tensor([1])
        L += torch.log(torch.exp(-d_h_pos)/denominator)
    return L

# Convert the models output to a logit matrix
def decode(z, pos_edge_index, neg_edge_index):
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #logits = torch.abs((z[edge_index[0]] * z[edge_index[1]])).sum(dim=1)
    #logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    #logits = (z[edge_index[0]] - z[edge_index[1]] + 0.01).norm(dim=-1)
    logits = hyperbolic_logits(z, pos_edge_index, neg_edge_index)
    return logits

def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] where the number of ones is equal to the length of 
    # pos_edge_index and the number of zeros is equal to the length of neg_edge_index
    
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    #E = pos_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# One epoch of training
def train(model, optimizer, train_data):
    model.train()
    
    z = model.forward(train_data.x, train_data.pos_edge_label_index) #forward
    for i in range(z.size(0)):
        print(torch.norm(z[i]))
        if torch.norm(z[i]) > 1:
            z[i] = z[i]/torch.norm(z[i])
    link_logits = decode(z, train_data.pos_edge_label_index, train_data.neg_edge_label_index) # decode
    
    link_labels = get_link_labels(train_data.pos_edge_label_index, train_data.neg_edge_label_index)
    #logits = hyperbolic_logits(z, train_data.pos_edge_label_index, train_data.neg_edge_label_index)
    #loss = nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss = poincare_loss(z, train_data.pos_edge_label_index, train_data.neg_edge_label_index)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, z

def test(model, train_data, test_data, val_data=None):
    model.eval()
    #x = torch.concat([train_data.x, test_data.x], dim=0)
    x = train_data.x
    pos_edge_index = torch.cat([train_data.pos_edge_label_index, test_data.pos_edge_label_index], dim=1)
    neg_edge_index = torch.cat([train_data.neg_edge_label_index, test_data.neg_edge_label_index], dim=1)
    if val_data is not None:
        pos_edge_index = torch.cat([pos_edge_index, val_data.pos_edge_label_index], dim=1)
        neg_edge_index = torch.cat([neg_edge_index, val_data.neg_edge_label_index], dim=1)

    with torch.no_grad():
        z = model.forward(x, pos_edge_index)
        link_logits = decode(z, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    return loss, z

# Train a model on a dataset, and return the loss
def train_model(model, optimizer, train_data, test_data, val_data, epochs=100, epoch_step=100):
    
    train_loss = []
    val_loss = []
    
    z_train = []

    # print(f' Epoch | Train loss | Val loss')
    # print(f'-------|------------|-----------')
    for epoch in range(epochs):
        loss, z = train(model, optimizer, train_data)
        if epoch % epoch_step == 0:
            train_loss.append(loss)
            z_train.append(z)
            val_loss.append(test(model, train_data, val_data)[0])
        # print(f'----{epoch:04}----|---{train_loss[-1]:.3f}---|---{val_loss[-1]:.3f}---')
    # print('')
    test_loss, z_test = test(model, train_data, test_data)
    return {"epochs":range(1, epochs+1, epoch_step), "train loss":train_loss, "val loss":val_loss, "test loss":test_loss, "z_train":z_train, "z_test":z_test}

def CN(PS, test):
    ## Use Common-Neighbours for Precision-Recall 
    PS_train = PS.copy()
    test_edges = np.transpose(test.pos_edge_label_index)
    PS_train.delete_edges([(x[0], x[1]) for x in test_edges])
    PR_CN = na.precision_recall_snapshots(PS_train, PS, metric='CN', step=1, plot=False)
    PR_CN['label'] = 'CN'
    return PR_CN

def LaBNE(PS, test):
    ## Generate LaBNE embedding and Precision-Recall on it
    PS_train = PS.copy()
    test_edges = np.transpose(test.pos_edge_label_index)
    PS_train.delete_edges([(x[0], x[1]) for x in test_edges])
    PR_LaBNE = na.precision_recall_snapshots(PS_train, PS, metric='LaBNE', step=1, plot=False)
    PR_LaBNE['label'] = 'LaBNE'
    return PR_LaBNE

def GraphSAGE(data, train, test, val, epochs):
    ## GraphSAGE model and PR on it
    graphSAGE_model = pyg.nn.GraphSAGE(in_channels=data.num_features, hidden_channels=32, out_channels=2, num_layers=3, dropout=0.1, act=torch.nn.Sigmoid(), jk='cat', bias=False)
    optimizer = torch.optim.SGD(graphSAGE_model.parameters(), lr=0.01)

    print(f'Training model: {graphSAGE_model} for {epochs} epochs')
    loss = train_model(model=graphSAGE_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
    print("Test loss:", loss['test loss'])
    R_SAGE, P_SAGE, predictions = LinkPrediction.precision_recall_trained_model(model=graphSAGE_model, train_data=train, test_data=test)
    _ = {'recall': R_SAGE, 'precision': P_SAGE, 'label': 'GraphSAGE', 'losses': loss, 'z_train': loss['z_train'], 'z_test': loss['z_test']}
    return _

def PNA(data, train, test, val, epochs, **kargs):
    ## PNA model and PR on it
    deg = torch.Tensor(pyg.utils.degree(train.edge_index[0], train.num_nodes))
    pna_model = pyg.nn.PNA(in_channels=data.num_features, hidden_channels=32, out_channels=2, num_layers=3, aggregators=['mean', 'max', 'sum', 'std'], scalers=['identity', 'linear'], deg=deg)
    optimizer = torch.optim.SGD(pna_model.parameters(), lr=0.01)

    print(f'Training model: {pna_model} for {epochs} epochs')
    loss = train_model(model=pna_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
    print("Test loss:", loss['test loss'])
    R_PNA, P_PNA, predictions = LinkPrediction.precision_recall_trained_model(model=pna_model, train_data=train, test_data=test)
    _ = {'recall': R_PNA, 'precision': P_PNA, 'label': 'PNA', 'losses': loss}
    return _

### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure

class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position

# class Shallownet(nn.Module):
#     r"""
#     A simple shallow (one hidden layer) neural network.
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Shallownet, self).__init__()
#         self.hidden = nn.Linear(input_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, output_dim)
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.tanh(self.hidden(x))
#         out = self.sigmoid(self.output(x))
#         return out

# class GCN(nn.Module):
#     r"""
#     A simple GCN layer.
#     """
#     def __init__(self, in_channels, out_channels, **kargs):
#         super(GCN, self).__init__()
#         self.gcn1 = pyg_nn.GCNConv(in_channels, out_channels, **kargs)
#         self.gcn2 = pyg_nn.GCNConv(out_channels, out_channels, **kargs)
        
#     def forward(self, data):
#         x = self.gcn1(data.x, data.pos_edge_label_index)
#         x = x.relu()
#         x = self.gcn2(x, data.pos_edge_label_index)
#         return x
    
#     def decode(self, z, pos_edge_index, neg_edge_index):
#         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
#         logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#         return logits

# class SAGE(nn.Module):
#     r"""
#     GraphSAGE model
#     """
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, num_layers=3, **kargs):
#         super(GraphSAGE, self).__init__()
#         self.dropout = dropout
#         self.conv_layers = nn.ModuleList()
#         self.conv_layers.append(pyg_nn.SAGEConv(in_channels, hidden_channels))
#         for _ in range(num_layers-2):
#             self.conv_layers.append(pyg_nn.SAGEConv(hidden_channels, hidden_channels))
#         self.conv_layers.append(pyg_nn.SAGEConv(hidden_channels, out_channels))
    
#     # reset all layer parameters
#     def reset_parameters(self):
#         for layer in self.conv_layers:
#             layer.reset_parameters()

#     def forward(self, data):
#         x = self.conv_layers[0](data.x, data.pos_edge_label_index)
#         x = x.relu()
#         for layer in self.conv_layers[1:-1]:
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = layer(x, data.pos_edge_label_index)
#             x = x.relu()
#         x = self.conv_layers[-1](x, data.pos_edge_label_index)
#         return torch.log_softmax(x, dim=-1)

#     def decode(self, z, pos_edge_index, neg_edge_index):
#         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
#         logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#         return logits