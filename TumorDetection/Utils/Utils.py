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
    return list(map(lambda x: fun(x, **kwargs), lst))


def tup2graph(tup, img_idx, mask_idx=None, dilations=(1, 2, 6), device='cpu'):
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
    assert all([d > 0 for d in dilations]), f'dilations must be all grater than 0. Got {dilations}'
    image = tup[img_idx]
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    x = torch.as_tensor(np.vstack(image), dtype=torch.uint8)
    if mask_idx is not None:
        y = torch.as_tensor(np.vstack(tup[mask_idx]), dtype=torch.uint8)
    else:
        y = None

    pos = torch.as_tensor([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])],
                          dtype=torch.int16)
    graph = Data(
        x=x,
        y=y,
        edge_index=None,
        pos=pos
    ).to(device=device)

    for d in dilations:
        dst = [torch.where(
            ((pos[:, 0] == pos[i, 0]) & (abs(pos[:, 1] - pos[i, 1]) == d)) |
            ((abs(pos[:, 0] - pos[i, 0]) == d) & (pos[:, 1] == pos[i, 1])) |
            ((abs(pos[:, 0] - pos[i, 0]) == d) & (abs(pos[:, 1] - pos[i, 1]) == d))
        )[0]
                 for i in range(graph.num_nodes)]

        srcdst = [[torch.tensor([i] * len(dst_i)), dst_i] for i, dst_i in enumerate(dst)]
        edge_index = torch.cat([torch.stack([src, dst]) for src, dst in srcdst], dim=1)
        # src = torch.tensor([i for i in range(graph.num_nodes)]).flatten()
        # for src_i in src:
        #     dst_i = torch.where(
        #         ((pos[:, 0] == pos[src_i, 0]) & (abs(pos[:, 1] - pos[src_i, 1]) == d)) |
        #         ((abs(pos[:, 0] - pos[src_i, 0]) == d) & (pos[:, 1] == pos[src_i, 1])) |
        #         ((abs(pos[:, 0] - pos[src_i, 0]) == d) & (abs(pos[:, 1] - pos[src_i, 1]) == d))
        #     )[0]
        #     edge_index_i = torch.stack([
        #         torch.tensor([src_i] * len(dst_i)).flatten(),
        #         dst_i
        #     ])
        #     graph.edge_index = (
        #         edge_index_i if graph.edge_index is None else
        #         torch.cat(tensors=[graph.edge_index, edge_index_i],
        #                   dim=1))
        graph.edge_index = (
            edge_index if graph.edge_index is None else
            torch.cat(tensors=[graph.edge_index, edge_index],
                      dim=1))

    graph.edge_index = tg.utils.to_undirected(edge_index=graph.edge_index)
    graph.edge_index, _ = tg.utils.remove_self_loops(edge_index=graph.edge_index)
    graph.coalesce()

    graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    return graph

##
# dst_i = [torch.where(
#     ((pos[:, 0] == pos[i, 0]) & (abs(pos[:, 1] - pos[i, 1]) == d)) |
#     ((abs(pos[:, 0] - pos[i, 0]) == d) & (pos[:, 1] == pos[i, 1])) |
#     ((abs(pos[:, 0] - pos[i, 0]) == d) & (abs(pos[:, 1] - pos[i, 1]) == d))
# )[0]
# for i in range(num_nodes)]
# srcdst=[[torch.tensor([i]*len(dst)), dst] for i, dst in enumerate(dst_i)]
# edge_index = torch.cat([torch.stack([a,b]) for a,b in srcdst[:10]])