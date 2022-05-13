import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from src import network_analysis as na
from src import LinkPrediction

device = 'cpu'

# Convert the models output to a logit matrix
def decode(z, pos_edge_index, neg_edge_index):
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return logits

def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] where the number of ones is equal to the length of 
    # pos_edge_index and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# One epoch of training
def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    
    z = model.forward(train_data.x, train_data.pos_edge_label_index) #forward
    link_logits = decode(z, train_data.pos_edge_label_index, train_data.neg_edge_label_index) # decode
    
    link_labels = get_link_labels(train_data.pos_edge_label_index, train_data.neg_edge_label_index)
    loss = nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    #probs = z @ z.T
    #probs = nn.functional.normalize(torch.concat((probs[train_data.pos_edge_label_index[0], train_data.pos_edge_label_index[1]],
                                                  #probs[train_data.neg_edge_label_index[0], train_data.neg_edge_label_index[1]])), dim=0)
    #loss = nn.functional.binary_cross_entropy(probs, link_labels)
    loss.backward()
    optimizer.step()

    return loss

# Train a model on a dataset, and return the loss
def train_model(model, optimizer, train_data, test_data, val_data, epochs=100):
    best_val_perf = test_perf = 0
    for epoch in range(1, epochs+1):
        train_loss = train(model, optimizer, train_data)
        print(f'{train_loss:.2f}...', end='')
    print('')
    return train_loss

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

def GraphSAGE(data, train, test, val, epochs, **kargs):
    ## GraphSAGE model and PR on it
    graphSAGE_model = pyg.nn.GraphSAGE(in_channels=data.num_features, hidden_channels=32, out_channels=16, num_layers=5)
    optimizer = torch.optim.SGD(graphSAGE_model.parameters(), lr=0.001)

    print(f'Training model: {graphSAGE_model} for {epochs} epochs')
    loss = train_model(model=graphSAGE_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
    print(f'Train loss: {loss}')
    R_SAGE, P_SAGE, predictions = LinkPrediction.PrecisionRecallTrainedModel(model=graphSAGE_model, train_data=train, test_data=test)
    _ = {'recall': R_SAGE, 'precision': P_SAGE, 'label': 'GraphSAGE'}
    return _

def PNA(data, train, test, val, epochs, **kargs):
    ## PNA model and PR on it
    deg = torch.Tensor(pyg.utils.degree(train.edge_index[0], train.num_nodes))
    pna_model = pyg.nn.PNA(in_channels=data.num_features, hidden_channels=32, out_channels=16, num_layers=3, aggregators=['mean', 'max', 'sum', 'std'], scalers=['identity', 'linear'], deg=deg)
    optimizer = torch.optim.SGD(pna_model.parameters(), lr=0.001)

    print(f'Training model: {pna_model} for {epochs} epochs')
    loss = train_model(model=pna_model, optimizer=optimizer, train_data=train, test_data=test, val_data=val, epochs=epochs)
    print(f'Train loss: {loss}')
    R_PNA, P_PNA, predictions = LinkPrediction.PrecisionRecallTrainedModel(model=pna_model, train_data=train, test_data=test)
    _ = {'recall': R_PNA, 'precision': P_PNA, 'label': 'PNA'}
    return _

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