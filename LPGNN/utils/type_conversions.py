def tensor_to_node_dict(tensor, node_names):
    """Convert a tensor to a dictionary of node names and values.
    
    Args:
        tensor (torch.Tensor): The tensor to convert.
        node_names (list): The names of the nodes.
        
    Returns:
        dict: The dictionary of node names and values.
    """
    node_dict = {}
    for i, node_name in enumerate(node_names):
        node_dict[node_name] = tensor[i]
    return node_dict