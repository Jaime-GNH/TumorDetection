import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric as tg


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


def tup2graph(tup, img_idx, mask_idx=None, dilations=(1, 3, 5, 7), device='cpu'):
    """
    Converts a tuple with an image to a graph.
    :param tup: tuple
    :param img_idx:
    :param mask_idx:
    :param dilations: (tup(int))
        Number of dilations.
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
    graph = Data(
        x=x,
        y=y,
        edge_index=torch.empty(),
        pos=pos
    ).to(device=device)

    for d in dilations:
        (row, col) = tg.utils.grid(3 + d, 3 + d)
        graph.edge_index = torch.cat(
            tensors=(graph.edge_index,
                     torch.stack([row[::d], col[::d]])),
            dim=1
        )
        graph.remove_self_loops()
        graph.to_undirected()
        graph.coalesce()

    graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    return graph
