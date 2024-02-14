import numpy as np
import skimage
import cv2
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


def tup2superpixel(tup, image_idx, mask_idx=None,
                   n_segments=10, compactness=10, sigma=0):
    """
    https://github.com/aryan-at-ul/imgraph/blob/main/imgraph/data/make_graph.py
    Converts a tuple with an image to a superpixel graph.
    :param tup:
    :param image_idx:
    :param mask_idx:
    :param n_segments:
    :param compactness:
    :param sigma:
    :return:
    """
    image = tup[image_idx]
    mask = tup[mask_idx]
    if isinstance(image, np.ndarray):
        device = 'cpu'
        if image.dtype == 'uint8' or np.max(image) > 1:
            image = image.astype('int')
        # x = torch.as_tensor(image, dtype=torch.float32, device=device)
    elif isinstance(image, torch.Tensor):
        raise ValueError(f'image must be a np.ndarray or torch.Tensor. Got {type(image)}')
        device = image.device
        if image.max() > 1.:
            image = image / 255.
        x = image.to(torch.float32)
    else:
        raise ValueError(f'image must be a np.ndarray or torch.Tensor. Got {type(image)}')

    if mask is not None:
        if isinstance(mask, np.ndarray):
            y = mask
            # y = torch.as_tensor(mask, device=device).to(torch.long)
        else:
            y = mask.to(torch.long)
    else:
        y = None

    segments = skimage.segmentation.slic(image, n_segments=n_segments,
                                         compactness=compactness, sigma=sigma,
                                         channel_axis=None
                                         )
    print(segments)
    print('with mask')
    print(skimage.filters.sobel(image, mask=y))
    print('without mask')
    print(skimage.filters.sobel(image))
    rag_img = skimage.graph.rag_boundary(segments,
                                         edge_map=skimage.filters.sobel(image)
                                         )
    rag_mask = skimage.graph.rag_boundary(segments,
                                          edge_map=skimage.filters.sobel(y)
                                          )
    # rag_y = skimage.graph.rag_boundary(segments,
    #                                    edge_map=y
    #                                    ) if y is not None else None

    return rag_img, rag_mask


def tup2graph(tup, img_idx, mask_idx,
              dilations=(1, 2, 6), kernel_kind='star'):
    """
    Converts a tuple with an image to a graph.
    :param tup: (tuple)
        Tuple containing image, mask and additional information for graph
    :param img_idx: (int)
        Index for image to be transformed in tuple
    :param mask_idx: (int)
        Index of target mask in tuple
    :param dilations: ((tup(int), int), (1, 2, 6))
        Number of dilations
    :param kernel_kind: (str, 'star')
        Kernel type
    :return: (torch.data.Data)
        Homogeneous graph.
    """
    image = tup[img_idx]
    if isinstance(image, np.ndarray):
        device = 'cpu'
        if image.dtype == 'uint8' or np.max(image) > 1:
            image = image.astype('float') / 255.
        x = torch.as_tensor(np.vstack(image), dtype=torch.float32)
    elif isinstance(image, torch.Tensor):
        device = image.device
        if image.max() > 1.:
            image = image / 255.
        x = torch.unsqueeze(image.flatten(), -1)
    else:
        raise ValueError(f'image must be a np.ndarray or torch.Tensor. Got {type(image)}')

    if tup[mask_idx] is not None:
        if isinstance(tup[mask_idx], np.ndarray):
            y = torch.as_tensor(tup[mask_idx], device=device).to(torch.long)
        else:
            y = tup[mask_idx].flatten().to(torch.long)
    else:
        y = None

    pos = torch.as_tensor([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])],
                          dtype=torch.int16).to(device)
    pos_idx = torch.as_tensor(list(range(pos.size(dim=0))), dtype=torch.int32).to(device)
    if isinstance(dilations, int):
        dilations = (dilations,)
    assert all([d > 0 for d in dilations]), f'dilations must be all grater than 0. Got {dilations}'

    edge_index = None
    for d in dilations:
        kernel = build_edge_kernel(d, kernel_kind, device)
        for k in kernel:
            # apply_kernel(edge_index, direction, hyperpos, hyperpos_idx, device)
            edge_index = apply_kernel(edge_index, k, pos, pos_idx, device)
    edge_index = tg.utils.to_undirected(edge_index=edge_index)
    edge_index, _ = tg.utils.remove_self_loops(edge_index=edge_index)

    graph = Data(
        x=x,
        y=y,
        edge_index=edge_index.to(torch.int64),
        # pos=pos
    )

    graph.coalesce()

    # graph.info = [l for i, l in enumerate(tup) if i not in [img_idx, mask_idx]]
    # graph.original_shape = image.shape
    graph.to(device=device, non_blocking=True)
    return graph


def tup2hypergraph(tup, image_idx, current_solution_idx=None, mask_idx=None,
                   hypernode_patch_dim=32, kernel_kind='star'):
    """
    Converts a tuple with an image to a hypergraph.
    :param tup: (tuple)
        Tupple containing data.
    :param image_idx: (int)
        image to be transformed index in tuple.
    :param current_solution_idx: (int, None)
        Current solution index in tuple.
    :param mask_idx: (int, None)
        target mask index in tuple.
    :param hypernode_patch_dim: (int, 128)
        (dim, dim) patch for creating hypernodes.
    :param kernel_kind: (str, 'star')
        Kernel type
    :return: (torch.data.Data)
        Homogeneous graph.
    """
    image = tup[image_idx]
    current_solution = tup[current_solution_idx]
    mask = tup[mask_idx]
    if isinstance(image, np.ndarray):
        device = 'cpu'
        # if image.dtype == 'uint8' or np.max(image) > 1:
            # image = image.astype('float') / 255.
        x = torch.as_tensor(image, dtype=torch.uint8, device=device)
        if current_solution is None:
            current_solution = np.ones(image.shape)
        current_solution = torch.as_tensor(current_solution, dtype=torch.float32, device=device)
    elif isinstance(image, torch.Tensor):
        device = image.device
        # if image.max() > 1.:
        #     image = image / 255.
        x = image
        if current_solution is None:
            current_solution = torch.ones(image.size(), device=device)
        current_solution = current_solution.to(torch.float32)
    else:
        raise ValueError(f'image must be a np.ndarray or torch.Tensor. Got {type(image)}')

    if mask is not None:
        if isinstance(mask, np.ndarray):
            y = torch.as_tensor(mask, device=device).to(torch.long)
        else:
            y = mask.to(torch.long)
    else:
        y = None

    mask_hypernodes = (current_solution
                       .unfold(0, hypernode_patch_dim, hypernode_patch_dim)
                       .unfold(1, hypernode_patch_dim, hypernode_patch_dim)
                       .flatten(0, 1)
                       .flatten(1)
                       .max(1)).values

    # hypernodes_x = (x
    #                 .unfold(0, hypernode_patch_dim, hypernode_patch_dim)
    #                 .unfold(1, hypernode_patch_dim, hypernode_patch_dim)
    #                 .flatten(0, 1)
    #                 .flatten(1)).to(device=device)
    hypernodes_x = (x
                    .unfold(0, hypernode_patch_dim, hypernode_patch_dim)
                    .unfold(1, hypernode_patch_dim, hypernode_patch_dim).flatten(0, 1))
    if hypernode_patch_dim > 1:
        descs = []
        for hnx in hypernodes_x.cpu().numpy():
            descs.append(cv2.calcHist(hnx, [0], None, [hypernode_patch_dim//4], [0, 256]).flatten())
        hypernodes_x = torch.as_tensor(np.array(descs).astype('float32')/255., dtype=torch.float32, device=device)
    else:
        hypernodes_x = hypernodes_x.flatten(1).to(device=device, dtype=torch.float32)/255.
    y = (y
         .unfold(0, hypernode_patch_dim, hypernode_patch_dim)
         .unfold(1, hypernode_patch_dim, hypernode_patch_dim)
         .flatten(0, 1)
         .flatten(1)
         .max(1).values).to(device=device, dtype=torch.float)

    # pos = torch.as_tensor([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])],
    #                       dtype=torch.int16, device=device)

    hyperpos = torch.as_tensor([[i, j]
                                for i in range(image.shape[0] // hypernode_patch_dim)
                                for j in range(image.shape[1] // hypernode_patch_dim)],
                               dtype=torch.int16,
                               device=device)
    hyperpos_idx = torch.as_tensor(list(range(hyperpos.size(dim=0))), dtype=torch.int32, device=device)

    edge_index = None
    kernel = build_edge_kernel(1, kernel_kind, device)
    for direction in kernel:
        edge_index = apply_kernel(edge_index, direction, hyperpos, hyperpos_idx, device)
    edge_index = tg.utils.to_undirected(edge_index=edge_index)
    edge_index, _ = tg.utils.remove_self_loops(edge_index=edge_index)

    graph = Data(
        x=hypernodes_x,
        y=y,
        # pos=pos,
        # hyperpos=hyperpos,
        edge_index=edge_index.to(torch.int64),
        hypernode_patch_dim=hypernode_patch_dim
    )
    graph = graph.subgraph((mask_hypernodes > 0).nonzero().flatten().to(device=device)).clone()
    graph.coalesce()

    # graph.info = [l for i, l in enumerate(tup) if i not in [image_idx, mask_idx, mask_idx + 1]]
    graph.original_shape = image.shape
    return graph.to(device, non_blocking=True), mask_hypernodes


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
        return torch.as_tensor([[d, 0], [0, d]], device=device)
    elif kind == 'hex':
        return torch.as_tensor([[-d, d], [d, d]], device=device)
    elif kind == 'star':
        return torch.cat([build_edge_kernel(d, kind='corner', device=device),
                          build_edge_kernel(d, kind='hex', device=device)])
    elif kind == 'square':
        kernel = torch.as_tensor(sum([[[i, j], [-i, j], [i, -j], [-i, -j],
                                       [i, i], [i, -i], [-i, i], [-i, -i]] for i, j in
                                      zip(range(d + 1), range(d, -1, -1))], []), device=device).unique(dim=0)
        return torch.cat([kernel[:kernel.size(dim=0) // 2], kernel[kernel.size(dim=0) // 2:]])
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
    dy, dx = int((image_shape[0]) / num_hops), int((image_shape[1]) / num_hops)
    diag = min(dy, dx)
    offset = max(dy, dx) - diag
    if kernel_kind == 'corner':
        dilations += [2 * diag, offset]
    elif kernel_kind == 'hex':
        dilations += [diag, 2 * offset]
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
