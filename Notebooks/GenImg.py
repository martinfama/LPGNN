from importlib.metadata import requires
import os
from math import ceil
from re import I
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
# import torchmetrics as thm
import LPGNN
import igraph as ig
import networkx as nx
import torch_geometric as pyg

import pyarrow as pa
import pyarrow.parquet as pq

import importlib
import powerlaw

import imageio

from tqdm import tqdm

def main(**kwargs):
    # Generate a PS network. Apply LaBNE. Set LaBNE positions
    # as initial positions for Poincaré Embedding. For each epoch
    # save a frame of the embedding. Save the frames as a gif.

    path = kwargs.get('path', 'figs/Embeddings/')
    folder = path + kwargs.get('folder', 'tmp/')
    output = path + kwargs.get('output', 'tmp.png')
    # if folder does not exist, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(path, folder, output)

    # Generate a PS network
    N = kwargs.get('N', 100)
    avg_k = kwargs.get('avg_k', 4)
    gma = kwargs.get('gma', 2.5)
    T = kwargs.get('T', 1)
    seed = kwargs.get('seed', 0)
    
    PS = LPGNN.popularity_similarity.generatePSNetwork(N, avg_k, gma, T, seed, dim=2)
    PS_nx = pyg.utils.to_networkx(PS, to_undirected=True)

    if kwargs.get('LaBNE'):
        PS_LaBNE = LPGNN.labne.generateLaBNE(PS, normalize_radius=0.1)
        init_pos = PS_LaBNE.LaBNE_node_positions
    else:
        init_pos = np.random.uniform(-0.01, 0.01, (N, 2))
        init_pos = th.from_numpy(init_pos).float()

    node_color = PS.node_polar_positions[:,1].detach().numpy()

    init_lr = kwargs.get('init_lr', 0.01)
    finish_lr = kwargs.get('finish_lr', 1)
    epochs = kwargs.get('epochs', 1000)
    epoch_order_mag = ceil(np.log10(epochs))
    r_order_interval = kwargs.get('r_order_interval', None)
    print(r_order_interval)
    
    lr = np.linspace(init_lr, finish_lr, epochs)
    
    for i in tqdm(range(epochs)):
    # for i in range(epochs):

        PS_Poincare = LPGNN.poincare_embedding.poincare_embedding(PS, epochs=int(20+lr[i]*10), lr=lr[i], expm='exact', init_pos=init_pos, r_ordering=False)
        init_pos = PS_Poincare.PoincareEmbedding_node_positions
        if r_order_interval is not None:
            if (i+1) % r_order_interval == 0:
                polar_pos = LPGNN.distances.to_spherical(init_pos)
                r_max = polar_pos[:, 0].max()
                r = LPGNN.labne.radial_ordering(PS, 'edge_index')
                # scale from [r.min(), r.max()] to [., r_max]
                r *= r_max/r.max()
                # reshape r to be [num_nodes, 1]
                polar_pos[:, 0] = r
                init_pos = LPGNN.distances.to_cartesian(polar_pos)

        pos_np = init_pos.detach().numpy()

    LPGNN.popularity_similarity.drawPSNetwork(PS, pos_name='node_positions', node_color=node_color, save_to=folder+'PS.png')
    LPGNN.popularity_similarity.drawPSNetwork(PS_LaBNE, pos_name='LaBNE_node_positions', node_color=node_color, save_to=folder+'PS_LaBNE.png')
    LPGNN.popularity_similarity.drawPSNetwork(PS_Poincare, pos_name='PoincareEmbedding_node_positions', node_color=node_color, save_to=folder+'PS_PE.png')

if __name__ == '__main__':
    # generate argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Generate a gif of the Poincaré Embedding of a PS network after LaBNE.')
    parser.add_argument('--N', type=int, default=300, help='Number of nodes in the network.')
    parser.add_argument('--avg_k', type=int, default=8, help='Average degree of the network.')
    parser.add_argument('--gma', type=float, default=2.5, help='Power law exponent of the network.')
    parser.add_argument('--T', type=float, default=0.1, help='Temperature of the network.')
    parser.add_argument('--seed', type=int, default=48, help='Seed for the network generation.')
    parser.add_argument('--LaBNE', action='store_true', default=False, help='Whether to use LaBNE to initialize the Poincaré Embedding.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run the Poincaré Embedding.')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate for the Poincaré Embedding.')
    parser.add_argument('--finish_lr', type=float, default=0.5, help='Final learning rate for the Poincaré Embedding.')
    parser.add_argument('--skip_generate', action='store_true', help='Skip the generation of the network and embedding.')
    parser.add_argument('--path', type=str, default='figs/Embeddings/', help='Path to the folder where the frames and gif will be saved.')
    parser.add_argument('--folder', type=str, default='tmp/', help='Folder where the frames will be saved.')
    parser.add_argument('--output', type=str, default='tmp.webp', help='Name of the gif file.')
    parser.add_argument('--r_order_interval', type=int, default=None, help='Interval between radial ordering of the nodes.')
    args = parser.parse_args()

    # run main
    main(**vars(args))