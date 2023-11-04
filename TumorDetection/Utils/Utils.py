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


def tup2graph(tup, img_idx, mask_idx=None,
              auto_dilations=False, num_hops=None,
              dilations=(1, 2, 6), kernel_kind='star',
              device='cpu'):
    """
    Converts a tuple with an image to a graph.
    :param tup: (tuple)
        Tuple containing image, mask and additional information for graph
    :param img_idx: (int)
        Index for image to be transformed in tuple
    :param mask_idx: (int, None)
        Index of target mask in tuple
    :param auto_dilations: (bool, False)
        Calculate dilations automatically
    :param num_hops: (int, None)
        Number of message passings. Only if auto_dilations
    :param dilations: ((tup(int), int), (1, 2, 6))
        Number of dilations. Only if not auto_dilations
    :param kernel_kind: (str, 'star')
        Kernel type
    :param device: (str, 'cpu')
        torch device
    :return: (torch.data.Data)
        Homogeneous graph.
    """
    image = tup[img_idx]
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    if image.dtype == 'uint8' or np.max(image) > 1:
        image = image.astype('float')/255.
    x = torch.as_tensor(np.vstack(image), dtype=torch.float32)
    if mask_idx is not None:
        y = torch.as_tensor(tup[mask_idx], dtype=torch.int8).flatten().unsqueeze(dim=1).to(device)
    else:
        y = None

    pos = torch.as_tensor([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])],
                          dtype=torch.int16).to(device)
    pos_idx = torch.as_tensor(list(range(pos.size(dim=0))), dtype=torch.int32).to(device)
    if not auto_dilations:
        if isinstance(dilations, int):
            dilations = (dilations,)
        assert all([d > 0 for d in dilations]), f'dilations must be all grater than 0. Got {dilations}'
    else:
        assert num_hops is not None and num_hops > 0,\
            f'If auto_dilations is True you must pass a positive num_hops. Got {num_hops}'
        dilations = calculate_dilations(image_shape=image.shape, num_hops=num_hops, kernel_kind=kernel_kind)

    edge_index = None
    for d in dilations:
        kernel = build_edge_kernel(d, kernel_kind, device)
        for k in kernel:
            edge_index = apply_kernel(edge_index, k, pos, pos_idx, device)
    edge_index = tg.utils.to_undirected(edge_index=edge_index)
    edge_index, _ = tg.utils.remove_self_loops(edge_index=edge_index)

    graph = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        pos=pos
    )

    graph.coalesce()

    graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    graph.original_shape = image.shape
    graph.to(device=device, non_blocking=True)
    return graph


def build_edge_kernel(d, kind='star', device='cpu'):
    """
    Returns a pixel to pixel kernel edge
    :param d: (int)
        dilation
    :param kind: (str, 'star')
        kernel kind
    :param device: (str, 'cpu')
        device
    :return: (torch.tensor)
        Kernel
    """
    if kind == 'corner':
        return torch.as_tensor([[d, 0], [0, d]]).to(device)
    elif kind == 'hex':
        return torch.as_tensor([[-d, d], [d, d]]).to(device)
    elif kind == 'star':
        return torch.cat([build_edge_kernel(d, kind='corner', device=device),
                          build_edge_kernel(d, kind='hex', device=device)])
    elif kind == 'square':
        kernel = torch.as_tensor(sum([[[i, j], [-i, j], [i, -j], [-i, -j],
                                       [i, i], [i, -i], [-i, i], [-i, -i]] for i, j in
                                      zip(range(d + 1), range(d, -1, -1))], [])).unique(dim=0).to(device)
        return torch.cat([kernel[:kernel.size(dim=0)//2], kernel[kernel.size(dim=0)//2:]])
    else:
        raise ValueError(f'value {kind} for kernel kind is not valid.')


def apply_kernel(edge_index, kernel, pos, pos_idx, device):
    """
    Apply a given kernel over a graph
    :param edge_index: (torch.tensor)
        Previous edge_index
    :param kernel: (torch.tensor)
        Kernel to apply
    :param pos: (torch.tensor)
        Positions of nodes
    :param pos_idx: (torch.tensor)
        Position index
    :param device: (str)
        Device to use
    :return: (torch.tensor)
        New egde_index
    """
    new_pos = pos + kernel
    _, idx, counts = torch.cat([pos, new_pos], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    edge_index_new = torch.stack([
        pos_idx[mask[:len(pos)]],
        pos_idx[mask[len(pos):]]
    ]).to(device)
    return edge_index_new if edge_index is None else torch.cat(
        tensors=[edge_index, edge_index_new], dim=1)


def calculate_dilations(image_shape, num_hops, kernel_kind):
    """
    Given an image_shape and a desired number of hops it calculates the necessary dilations to fill the space.
    This method ensures percolation.
    Applying this method to all images for training could incurr in loosing meaning.
    This is a method to study the global dilations to apply.
    :param image_shape: (tuple(int))
        Image original shape
    :param num_hops: (int)
        Number of message passing
    :param kernel_kind: (str)
        Kernel type to calculate
    :return: tup(int)
        Dilations with len between 1 an 3
    """
    dilations = [1]
    dy, dx = int((image_shape[0])/num_hops), int((image_shape[1])/num_hops)
    diag = min(dy, dx)
    offset = max(dy, dx) - diag
    if kernel_kind == 'corner':
        dilations += [2*diag, offset]
    elif kernel_kind == 'hex':
        dilations += [diag, 2*offset]
    else:
        dilations += [diag, offset]
    return tuple(sorted(set([d for d in dilations if d > 0])))


def graph_to_image_mask(graph):
    """
    Given a graph it returns a gray_scale image
    :param graph:
    :return:
    """
    x = graph.x[:, -1].unflatten(graph.original_shape[:2]).cpu().numpy()
    y = graph.y
    if y is not None:
        y = y.unflatten(graph.original_shape[:2]).cpu().numpy()

    return x, y
