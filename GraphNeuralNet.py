import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

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
    return train_loss

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