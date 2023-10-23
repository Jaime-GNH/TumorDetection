import numpy as np
import torch
from torch_geometric.data import Data


def apply_function2list(lst, fun, **kwargs):
    """
    Apply a function (fun) to each element in list (lst) using kwargs.
    :param lst: (list(np.nd.array))
        List to be used
    :param fun: (object)
        Function to be applied where first param accepts an element of lst
    :param kwargs: (dict)
        Kwargs to be used in the function
    :return: (list)
        Transformed result
    """
    # TODO: Add multiprocessing ORDERED
    return list(map(lambda x: fun(x, **kwargs), lst))


def tup2graph(tup, img_idx, mask_idx=None, device='cpu'):
    """
    Converts a tuple with an image to a graph.
    :param tup: tuple
    :param img_idx:
    :param mask_idx:
    :param device:
    :return: graph
    """
    image = tup[img_idx]
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    x = torch.as_tensor(np.vstack(image), dtype=torch.uint8)
    if mask_idx is not None:
        y = torch.as_tensor(np.vstack(tup[mask_idx]), dtype=torch.uint8)
    else:
        y = None

    pos = torch.as_tensor([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])])
    # TODO: Edge_index as [2, num_edges]
    #  [0, edge] -> src
    #  [1, dst] -> dst
    #  Not possible if all connected (
    #       src == num_nodes**2 and dst == num_nodes**2
    #  )
    #  Check by neighbouring and dilations

    graph = Data(
        x=x,
        y=y,
        pos=pos
    ).to(device=device)
    graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    return graph
