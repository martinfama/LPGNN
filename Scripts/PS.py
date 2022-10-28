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

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from LPGNN.utils import get_ram

dir_name = 'data/Distance_Lists/PS/PyG/'

print('STARTING ASI EMBEDDING PROTOCOL...')
get_ram()
print('Loading ASI network...')

# load ASI with pickle
with open('Network_Files/ASI_pyg.pkl', 'rb') as f:
    ASI = pickle.load(f)

print('ASI network loaded')
print('Initializing embedding search...')
get_ram(strt='RAM: ')

DIMs = [3,4,5,6,7,8,9,10,15,20]
total_epochs = [3*10**5, 6*10**5, 10**6]
LR = [0.01, 0.1, 0.001, 0.0001]
EXPM = ['exact', 'approx']
R_Order = [False, True]

print('Hyperparameters:')
print('DIMs:', DIMs)
print('total_epochs:', total_epochs)
print('LR:', LR)
print('expm:', EXPM)
print('r_order:', R_Order)

for lr in LR:
    print(' LR:', lr)
    for t_epochs in total_epochs:
            for expm in EXPM:
                print('   expm:', expm)
                for r_order in R_Order:
                    if r_order:
                        r_order = int(t_epochs*0.3)
                    print('         r_order:', r_order)
                    for dim in DIMs:
                        print('      DIM:', dim)
                        
                        labne_pyg_fname = dir_name+f'ASI_LaBNE_pyg_dim_{dim}'
                        if not os.path.exists(labne_pyg_fname):
                            print('         Building LaBNE...')
                            # generate LaBNE in dim dimensions
                            ASI_LaBNE = LPGNN.labne.generateLaBNE(ASI, edge_index='edge_index', normalize_radius=False, dim=dim)
                            print('         LaBNE built...')
                            get_ram(strt='         RAM: ')
                            with open(labne_pyg_fname, 'wb') as f:
                                pickle.dump(ASI_LaBNE, f)
                        else:
                            print('         Loading LaBNE...')
                            with open(labne_pyg_fname, 'rb') as f:
                                ASI_LaBNE = pickle.load(f)
                            get_ram(strt='         RAM: ')
                    
                        pe_pyg_fname = dir_name+f'ASI_PE_pyg_dim_{dim}_lr_{lr}_epochs_{t_epochs}_expm_{expm}_r_order_{r_order}'
                        if not os.path.exists(pe_pyg_fname):
                            print(f'         Generating {pe_pyg_fname}...')
                            # set maximum radius to 0.1 for Poincare
                            r = ASI_LaBNE.LaBNE_node_polar_positions[:,0] / ASI_LaBNE.LaBNE_node_polar_positions[:,0].max() * 0.1
                            r = r.reshape(-1,1)
                            theta = ASI_LaBNE.LaBNE_node_polar_positions[:,1:]
                            # init_pos for Poincare Embedding
                            init_pos = LPGNN.distances.to_cartesian(th.cat((r, theta), dim=1))

                            print('         Starting PE:')
                            ASI_Poincare = LPGNN.poincare_embedding.poincare_embedding(ASI_LaBNE, epochs=t_epochs, lr=lr, expm=expm, init_pos=init_pos, epochs_per_r_order=r_order, dim=dim)
                            print('         PE complete')
                            f = open(pe_pyg_fname, 'wb')
                            pickle.dump(ASI_Poincare, f)
                            get_ram(strt='         RAM: ')
                        else:
                            print(f'         Skipping {pe_pyg_fname}...')