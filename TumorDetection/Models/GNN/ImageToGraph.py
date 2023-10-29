from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import ImageToGraphCall
from TumorDetection.Utils.Utils import apply_function2list, tup2graph


class ImageToGraph(BaseClass):
    """
    Image (np.ndarray) to graph (torch_geometric.Data) converter
    """
    def __init__(self):
        """
        Class constructor.
        """
        pass

    def __call__(self, data, **kwargs):
        """
        Main function.
        :param data: list(tuple)
        :param kwargs:
        :return: list(torch_geometric.Data)
        """
        kwargs = self._default_config(ImageToGraphCall, **kwargs)
        self.train = kwargs.get('train')
        images_tup_idx = kwargs.get('images_tup_idx')
        mask_tup_idx = kwargs.get('mask_tup_idx')
        device = kwargs.get('device')
        dilations = kwargs.get('dilations')
        if isinstance(dilations, int):
            dilations = (dilations, )
        graphs = apply_function2list(
            data,
            tup2graph,
            dilations=dilations,
            img_idx=images_tup_idx,
            mask_idx=mask_tup_idx if self.train else None,
            device=device
        )
        return graphs
