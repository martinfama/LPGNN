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
    
    z = model.forward(train_data.x, train_data.pos_edge_label_index, train_data.edge_type[:train_data.pos_edge_label_index.shape[1]]) #forward
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