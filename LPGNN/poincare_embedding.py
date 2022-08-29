import numpy as np
import torch as th
import torch_geometric as pyg
from gensim.models.poincare import PoincareModel

from .embedding import *

@embedding
def poincare_embedding(data:pyg.data.Data, epochs=100, only_coordinates=False, **kwargs):
    """ Apply the Poincaré embedding [1] to the given graph.

        [1] Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. arXiv. https://doi.org/10.48550/ARXIV.1705.08039 

    Args:
        data (pyg.data.Data): The graph to embed.
    Returns:
        pyg.data.Data: The graph with the embedding.
    """
    
    model = PoincareModel(data.edge_index.T.detach().numpy(), **kwargs)
    model.train(epochs)

    embeddings = np.array([model.kv[node] for node in range(data.num_nodes)])
    
    if only_coordinates:
        return embeddings
    
    data_Poincare = data.clone()
    data_Poincare.PoincareEmbedding_node_positions = th.Tensor(embeddings)
    return data_Poincare
