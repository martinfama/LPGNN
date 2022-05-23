"""
Generate animations of attribute passing in a GNN. This is seen
as a coloring/resizing of the nodes in the graph, with respect to
the attributes of the nodes. As the GNN operates on a graph, these
attributes are updated, and the graph is then rendered.
"""
import os
import imageio
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import LPGNN.utils.type_conversions

def draw_graph_with_attrs(G, attrs, **kargs):
    """Draws a graph, coloring the nodes according to the attributes.

    Args:
        G (networkx.Graph): The corresponding graph.
        attrs (torch.Tensor): The attributes of the nodes.

    Returns:
        None
    """

    # Convert the attributes to a dictionary.
    node_dict = LPGNN.utils.type_conversions.tensor_to_node_dict(attrs, G.nodes())
    
    # If not set, set the node position and size.
    if 'pos' not in G.graph:
        G.graph['pos'] = nx.spring_layout(G)
    if 'size' not in G.graph:
        G.graph['size'] = [20*val for (node, val) in G.degree()]

    fig, ax = plt.subplots(figsize=(12,10))

    # Draw the graph. The node colors are set according to the attributes.
    nodes = nx.draw_networkx_nodes(G, pos=node_dict, node_color=G.graph['color'],
                                   nodelist=node_dict.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, node_dict, alpha=0.5)
    
    cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(nodes, cax=cax, format='%.2f', ticks=[kargs['min_attr'], kargs['max_attr']])
    #if 'max_attr' in kargs and 'min_attr' in kargs:
    #    cbar.boundaries(kargs['min_attr'], kargs['max_attr'])
    plt.axis('off')
    plt.figaspect(8)
 
    return fig, ax, cax, nodes

def animate_graph_with_attrs(G, attrs_list, savename=None):
    """Generate an animation of the graph attributes updating. The attributes
    are passed as a list in which each entry is a torch.Tensor, indicating
    the attrs at that moment.

    Args:
        G (networkx.Graph): The corresponding graph.
        attrs_list (list): List of torch.Tensors
    """

    max_attr, min_attr = max([torch.max(m) for m in attrs_list]).detach().numpy(), min([torch.min(m) for m in attrs_list]).detach().numpy()

    fig, ax, cax, nodes = draw_graph_with_attrs(G, attrs_list[0].detach().numpy(), max_attr=max_attr, min_attr=min_attr)

    dir_path = os.path.join(os.getcwd(), 'figs/animations', savename)
    os.makedirs(dir_path) if not os.path.exists(dir_path) else None
    images = []
    fig.savefig(dir_path+f'/000.png')
    images.append(imageio.imread(dir_path+f'/000.png'))

    def update_colors(i):
        node_dict = LPGNN.utils.type_conversions.tensor_to_node_dict(attrs_list[i].detach().numpy(), G.nodes())
        # Update the node colors.
        ax.clear()
        pos = node_dict
        nx.draw_networkx_nodes(G, pos=node_dict, 
                                   node_color=G.graph['color'],
                                   nodelist=node_dict.keys(), ax=ax)
        edges = nx.draw_networkx_edges(G, node_dict, alpha=0.5, ax=ax)
        #nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
        return nodes,

    for i, attrs in enumerate(attrs_list[1:]):
        update_colors(i+1)
        fig.savefig(dir_path+f'/{i+1:03}.png')
        images.append(imageio.imread(dir_path+f'/{i+1:03}.png'))
    imageio.mimsave(dir_path+'/out.gif', images)

    return

def draw_embedding(G, embedding, **kargs):
    """Draws a graph, coloring the nodes according to the attributes.

    Args:
        G (networkx.Graph): The corresponding graph.
        attrs (torch.Tensor): The attributes of the nodes.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12,10))

    # Draw the graph. The node colors are set according to the attributes.
    nodes = nx.draw(G, pos=embedding, node_color=G.graph['color'], node_size=G.graph['size'])
    #nodes = nx.draw_networkx_nodes(G, pos=G.graph['pos'], node_color=G.graph['color'])
    #nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    #edges = nx.draw_networkx_edges(G, embedding, alpha=0.5)
    
    #if 'max_attr' in kargs and 'min_attr' in kargs:
    #    cbar.boundaries(kargs['min_attr'], kargs['max_attr'])
    plt.axis('off')
    plt.figaspect(8)
 
    return fig, ax