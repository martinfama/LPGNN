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

    path = kwargs.get('path', 'figs/animations/Poincare/')
    folder = path + kwargs.get('folder', 'tmp/')
    output = path + kwargs.get('output', 'tmp.webp')
    # if folder does not exist, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(path, folder, output)

    if not kwargs.get('skip_generate'):

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

        fig, ax = plt.subplots(figsize=(10,10))
        ax.grid()
        # remove x and y axis, ticks and labels
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        node_color = PS.node_polar_positions[:,1].detach().numpy()
        pos_np = init_pos.detach().numpy()
        edges = nx.draw_networkx_edges(PS_nx, pos=pos_np, ax=ax, alpha=0.2, width=0.5, edge_color='k')
        points = ax.scatter(pos_np[:,0], pos_np[:,1], s=30, c=node_color, cmap='hsv', zorder=100)

        init_lr = kwargs.get('init_lr', 0.01)
        finish_lr = kwargs.get('finish_lr', 1)
        epochs = kwargs.get('epochs', 1000)
        epoch_order_mag = ceil(np.log10(epochs))
        r_order_interval = kwargs.get('r_order_interval', None)
        print(r_order_interval)
        
        lr = np.linspace(init_lr, finish_lr, epochs)
        
        for i in tqdm(range(epochs)):
        # for i in range(epochs):

            fig.savefig(folder+f'{i:0{epoch_order_mag}d}.png', dpi=100, bbox_inches='tight')

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
            points.set_offsets(pos_np)
            # update the edges matplotlib.collections.LineCollection position
            edges.set_offsets(pos_np)
            # update edges.set_verts by getting the new positions of the pairs of nodes
            edges.set_verts([[(pos_np[e[0],0], pos_np[e[0],1]), (pos_np[e[1],0], pos_np[e[1],1])] for e in PS.edge_index.T])

    # generate a small size webp file with the images in the tmp folder

    import imageio
    from PIL import Image, ImageSequence

    # load all pngs from the tmp folder using PIL 
    images = []
    for filename in sorted(os.listdir(folder)):
        print(filename)
        if filename.endswith('.png'):
            img = Image.open(folder+filename)
            images.append(img)
    images[0].save(output, format='WEBP', save_all=True, append_images=images[1:], loop=0, duration=20)

    #import webp
    #Load a PIL image array from the specified .webp animation file
    #anim = webp.load_images(output)
    # Grab a reference to the first frame, and save the entire PIL image array as GIF with 70ms frames (14.286 FPS)
    # anim[0].save('output.gif', save_all=True, append_images=anim[0:], duration=70, loop=0)

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
    parser.add_argument('--path', type=str, default='figs/animations/Poincare/', help='Path to the folder where the frames and gif will be saved.')
    parser.add_argument('--folder', type=str, default='tmp/', help='Folder where the frames will be saved.')
    parser.add_argument('--output', type=str, default='tmp.webp', help='Name of the gif file.')
    parser.add_argument('--r_order_interval', type=int, default=None, help='Interval between radial ordering of the nodes.')
    args = parser.parse_args()

    # run main
    main(**vars(args))