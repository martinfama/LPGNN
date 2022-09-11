
import sys

import matplotlib as mpl
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

x = th.Tensor([0.0,1])
v = th.Tensor([0,10])
v = x + v

exp_x_v = LPGNN.poincare_embedding.exact_expm(x, v)
a_exp_x_v = LPGNN.poincare_embedding.approx_expm(x, v)

### ANIMATIONS

plt.ioff()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# draw a circle of radius 1 centered at (0,0)
circle = plt.Circle((0, 0), 1, color='k', fill=False, zorder=0)
ax.add_artist(circle)
# draw 8 line segments from the origin to the circle
# make the segments dotted and slightly transparent
for i in range(0, 360, 45):
    circle_arcs = ax.plot([0, np.cos(i*np.pi/180)], [0, np.sin(i*np.pi/180)], 'k--', alpha=0.5, zorder=0)

#draw a dot at x
x_line = ax.plot([x[0], v[0]], [x[1], v[1]], 'g-', zorder=1, label='x->v')
# draw exponential map of x, and approximate exponential map of x
a_exp_line = ax.plot([x[0], a_exp_x_v[0]], [x[1], a_exp_x_v[1]], 'r-', zorder=2, label='approx exp')
exp_line = ax.plot([x[0], exp_x_v[0]], [x[1], exp_x_v[1]], 'b-', zorder=3, label='exact exp')

ax.legend(loc='upper right')

def on_click(event):
    if event.button is mpl.backend_bases.MouseButton.LEFT:
        x[0] = event.xdata
        x[1] = event.ydata
    if event.button is mpl.backend_bases.MouseButton.RIGHT:
        v[0] = event.xdata
        v[1] = event.ydata

    exp_x_v = LPGNN.poincare_embedding.exact_expm(x, v-x)
    a_exp_x_v = LPGNN.poincare_embedding.approx_expm(x, v-x)

    x_line[0].set_data([x[0], v[0]], [x[1], v[1]])
    exp_line[0].set_data([x[0], exp_x_v[0]], [x[1], exp_x_v[1]])
    a_exp_line[0].set_data([x[0], a_exp_x_v[0]], [x[1], a_exp_x_v[1]])

    fig.canvas.draw()

plt.connect('button_press_event', on_click)
plt.show()