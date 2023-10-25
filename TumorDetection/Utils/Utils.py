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
        edge_index=None,
        pos=pos
    ).to(device=device)

    for d in dilations:
        edge_kernel = build_edge_kernel(d)
        if graph.edge_index is None:
            graph.edge_index = None
            # TODO:
        else:
            # TODO:
            pass
        graph.remove_self_loops()
        graph.to_undirected()
        graph.coalesce()

    graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    return graph


def build_edge_kernel(dilation):
    """
    Returns an edge_kernel with dilation
    :param dilation: (int)
        blocks to step over -1
    :return: (tensor)
        kernel.
    """
    (row, _), _ = tg.utils.grid(2 * dilation + 1, 2 * dilation + 1)
    dst = (
        row[torch.where(
            (row % (2 * dilation + 1) == 0) |
            ((row + 1) % (2 * dilation + 1) == 0) |
            ((row - dilation) % (2 * dilation + 1) == 0) &
            (row % 3 == 0)
        )].unique()
    )
    dst = dst[torch.where(dst % 3 == 0)]
    src = torch.tensor([row.max() // 2] * torch.numel(dst))
    return torch.stack([src, dst])


def apply_edge_kernel(graph):
    # TODO
    pass
