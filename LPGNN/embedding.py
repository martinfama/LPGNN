import numpy as np
import torch as th
import torch_geometric as pyg

# create a wrapper that will decorate functions that generate embeddigns
def embedding(embedding_function):
    def wrapper(*args, **kwargs):
        """ Apply the embedding to the given graph.
        
        Args:
            data (pyg.data.Data): The graph to embed.
        Returns:
            pyg.data.Data: The graph with the embedding.
        """
        embeddings = embedding_function(*args, **kwargs)
        if kwargs.get('only_coordinates'):
            print('Good one!')
            return embeddings
        
        return embeddings
    return wrapper