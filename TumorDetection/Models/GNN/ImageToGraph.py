from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import ImageToGraphCall
from TumorDetection.Utils.Utils import apply_function2list, tup2graph, tup2hypergraph


class ImageToGraph(BaseClass):
    """
    Image (np.ndarray) to graph (torch_geometric.Data) converter
    """
    def __call__(self, data, **kwargs):
        """
        Main function.
        :param data: (list(tuple), tuple)
        :param kwargs: ImageToGraphCall
        :return: list(torch_geometric.Data)
        """
        kwargs = self._default_config(ImageToGraphCall, **kwargs)
        mask = kwargs.get('mask')
        images_tup_idx = kwargs.get('images_tup_idx')
        mask_tup_idx = kwargs.get('mask_tup_idx')
        device = kwargs.get('device')
        kernel_kind = kwargs.get('kernel_kind')
        dilations = kwargs.get('dilations')
        hypergraph = kwargs.get('hypergraph')
        if hypergraph:
            assert len(data) > mask_tup_idx + 1 and (mask_tup_idx > images_tup_idx), \
                f'data must contain (img, mask, current_solution) in that order. Can be None but must exist.'
            if isinstance(data, tuple) and not isinstance(data[0], tuple):
                return tup2hypergraph(data,
                                      image_idx=images_tup_idx,
                                      current_solution_idx=mask_tup_idx+1,
                                      mask_idx=mask_tup_idx,
                                      hypernode_patch_div=kwargs.get('hypernode_patch_div'),
                                      kernel_kind=kernel_kind,
                                      device=device
                                      )
            elif isinstance(data, list) and isinstance(data[0], tuple):
                return apply_function2list(
                    data,
                    tup2hypergraph,
                    image_idx=images_tup_idx,
                    current_solution_idx=mask_tup_idx + 1,
                    mask_idx=mask_tup_idx,
                    hypernode_patch_dim=kwargs.get('hypernode_patch_dim'),
                    kernel_kind=kernel_kind,
                    device=device
                )
            else:
                raise ValueError(f'data must be list of tuples or single tuple. Got {type(data)}')
        else:
            if isinstance(data, tuple) and not isinstance(data[0], tuple):
                return tup2graph(data,
                                 img_idx=images_tup_idx,
                                 mask_idx=mask_tup_idx if mask else None,
                                 dilations=dilations,
                                 kernel_kind=kernel_kind,
                                 device=device
                                 )
            elif isinstance(data, list) and isinstance(data[0], tuple):
                return apply_function2list(
                    data,
                    tup2graph,
                    img_idx=images_tup_idx,
                    mask_idx=mask_tup_idx if mask else None,
                    dilations=dilations,
                    kernel_kind=kernel_kind,
                    device=device
                )
            else:
                raise ValueError(f'data must be list of tuples or single tuple. Got {type(data)}')
