from functools import total_ordering
import os, psutil
import importlib

import numpy as np
import pandas as pd
import torch as th
import networkx as nx
import torch_geometric as pyg

import LPGNN

import pickle

def get_ram():
    process = psutil.Process(os.getpid())
    print('\r', f'RAM usage: {process.memory_info().rss/10**6} MB', end=' ')

print('STARTING ASI EMBEDDING PROTOCOL...')
get_ram()
print('Loading ASI network...')

ASI = th.Tensor(np.loadtxt('Network_Files/ASI_t0.txt')).T.type(th.int64)
ASI = ASI-1 # Convert to 0-indexing
ASI = pyg.utils.to_undirected(ASI)
ASI = pyg.data.Data(edge_index=ASI)
ASI.num_nodes = ASI.num_nodes

# Second snapshot, which we consider to be the test set.
ASI_test = th.Tensor(np.loadtxt('Network_Files/ASI_t1.txt')).T.type(th.int64)
ASI_test = ASI_test-1

# remove edges in test set which are also in training set
ASI_test_temp = th.Tensor([[-1,-1]])
for edge in ASI_test.T:
    if len(th.where((edge == ASI.edge_index.T).all(dim=1))[0]) == 0:
        ASI_test_temp = th.cat((ASI_test_temp, edge.view(1,2)), dim=0)
ASI_test = ASI_test_temp[1:].T.long()
ASI.test_pos_edge_label_index = ASI_test

print('ASI network loaded')
print('Initializing embedding search...')
get_ram()

DIMs = [3,4,5,6,10,15,20]
epochs_per_radial_ordering = 1000
total_epochs = [10**5, 10**6, 10**7, 10**8]
LR = [0.01, 0.001, 0.0001]

print('Hyperparameters:')
print('DIMs:', DIMs)
print('epochs_per_radial_ordering:', epochs_per_radial_ordering)
print('total_epochs:', total_epochs)
print('LR:', LR)

for dim in DIMs:
    print('DIM:', dim)
    print('    Building LaBNE...')
    # generate LaBNE in dim dimensions
    ASI_LaBNE = LPGNN.labne.generateLaBNE(ASI, edge_index='edge_index', normalize_radius=False, dim=dim)
    print('    LaBNE built...')
    get_ram()
    print('    Generating PR curve...')
    # generate PR-curves
    ASI_R_LaBNE, ASI_P_LaBNE = LPGNN.LinkPrediction.precision_recall_score_file(ASI_LaBNE, position_name='LaBNE_node_positions', filename='data/Distance_Lists/ASI/ASI_LaBNE_hyp', chunk_size=200000, skip_file_prep=False, step_size=10000, dist='hyp')
    print('    PR curve generated...')
    get_ram()
    print('    Deleting distance list files...')
    os.remove('data/Distance_Lists/ASI/ASI_LaBNE_hyp')
    os.remove('data/Distance_Lists/ASI/ASI_LaBNE_hyp_sorted')
    print('    Distance list files deleted...')
    get_ram()
    with open(f'data/Distance_Lists/ASI/ASI_LaBNE_hyp_dim_{dim}.pickle', 'wb') as f:
            pickle.dump([ASI_R_LaBNE, ASI_P_LaBNE], f)

    for lr in LR:
        print('      LR:', lr)
        for t_epochs in total_epochs:
            # set maximum radius to 0.1 for Poincare
            r = ASI_LaBNE.LaBNE_node_polar_positions[:,0] / ASI_LaBNE.LaBNE_node_polar_positions[:,0].max() * 0.1
            r = r.reshape(-1,1)
            theta = ASI_LaBNE.LaBNE_node_polar_positions[:,1:]
            # init_pos for Poincare Embedding
            init_pos = LPGNN.distances.to_cartesian(th.cat((r, theta), dim=1))

            print('        Starting PE:')
            ASI_Poincare = LPGNN.poincare_embedding.poincare_embedding(ASI_LaBNE, epochs=t_epochs, lr=lr, expm='exact', init_pos=init_pos, r_ordering=epochs_per_radial_ordering, dim=dim)
            print('        PE complete')

            print('        Generating PR curve...')
            ASI_R_Poincare, ASI_P_Poincare = LPGNN.LinkPrediction.precision_recall_score_file(ASI_Poincare, position_name='PoincareEmbedding_node_positions', filename='data/Distance_Lists/ASI/ASI_Poincare_hyp', chunk_size=200000, skip_file_prep=False, step_size=10000, dist='hyp')
            print('        PR curve generated...')
            get_ram()
            print('        Deleting distance list files...')
            os.remove('data/Distance_Lists/ASI/ASI_Poincare_hyp')
            os.remove('data/Distance_Lists/ASI/ASI_Poincare_hyp_sorted')
            print('        Distance list files deleted...')
            get_ram()

            with open(f'data/Distance_Lists/ASI/ASI_Poincare_hyp_dim_{dim}_lr_{lr}_epochs_{t_epochs}.pickle', 'wb') as f:
                pickle.dump([ASI_R_Poincare, ASI_P_Poincare], f)

            