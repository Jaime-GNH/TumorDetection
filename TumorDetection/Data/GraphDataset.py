import math
from torch.utils.data import Dataset

from TumorDetection.Data.Loader import DataPathLoader, ImageLoader
from TumorDetection.Data.Preprocess import Preprocessor
from TumorDetection.Models.GNN.ImageToGraph import ImageToGraph
from TumorDetection.Utils.DictClasses import GraphDatasetInit

from TumorDetection.Utils.BaseClass import BaseClass


class GraphDataset(Dataset, BaseClass):
    """
    """
    def __init__(self, root_dir, **kwargs):
        """
        Class constructor
        :param root_dir: directory with files.
        :param kwargs: DataPathLoaderCall Class
        """
        super().__init__()
        kwargs = self._default_config(GraphDatasetInit, **kwargs)
        if kwargs.get('inference'):
            kwargs.update({'train': False})
        self.root_dir = root_dir
        dp = DataPathLoader(dir_path=root_dir)
        self.paths_classes = dp(**kwargs.get('datapathloader_transforms'))
        if kwargs.get('train'):
            self.paths_classes = self.paths_classes[:-math.floor(len(self)*kwargs.get('train_test_split'))]
        elif not kwargs.get('inference') and not kwargs.get('train'):
            self.paths_classes = self.paths_classes[-math.floor(len(self) * kwargs.get('train_test_split')):]
        self.image_loader = ImageLoader()
        self.image_loader_transforms = kwargs.get('imageloader_transforms')
        self.preprocessor = Preprocessor()
        self.preprocessor_transforms = kwargs.get('preprocessor_transforms')
        self.image2graph = ImageToGraph()
        self.image2graph_transforms = kwargs.get('image2graph_transforms')

    def __len__(self):
        """
        Length of the dataset
        :return: (int)
            length
        """
        return len(self.paths_classes)

    def __getitem__(self, item):
        """
        Returns torch_geometric.Data corresponding to image.
        :param item: (image)
        :return: torch_geometric.Data
        """
        item_load = self.image_loader(self.paths_classes[item], **self.image_loader_transforms)
        item_prep = self.preprocessor([item_load[2]], **self.preprocessor_transforms)
        new_item = (item_load[0], item_load[1], item_prep[0], item_load[3])
        return self.image2graph(new_item, **self.image2graph_transforms)
