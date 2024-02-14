import math
import random

import cv2
import numpy as np
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf


from TumorDetection.Data.Loader import DataPathLoader, ImageLoader
from TumorDetection.Data.Preprocess import Preprocessor
from TumorDetection.Models.GNN.ImageToGraph import ImageToGraph
from TumorDetection.Utils.DictClasses import (GraphDatasetInit, GraphDataLoaderInit,
                                              ImageDatasetInit, ImageDataLoaderInit)

from TumorDetection.Utils.BaseClass import BaseClass


class ListCollater:
    def __init__(self, dataset, follow_batch=None, exclude_keys=None):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, list):
            return Batch.from_data_list(
                sum(batch, []),
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch):
        return self(batch)


class GraphDataset(Dataset, BaseClass):
    """
    """

    def __init__(self, root_dir, mode, batch_size, **kwargs):
        """
        Class constructor
        :param root_dir: directory with files.
        :param kwargs: DataPathLoaderCall Class
        """
        super().__init__()
        kwargs = self._default_config(GraphDatasetInit, **kwargs)
        # if mode == 'inference':
        #     kwargs.update({'train': False})
        self.root_dir = root_dir
        dp = DataPathLoader(dir_path=root_dir)
        self.paths_classes = dp(**kwargs.get('datapathloader_transforms'))
        random.Random(kwargs.get('seed')).shuffle(self.paths_classes)
        if mode == 'train':
            self.paths_classes = self.paths_classes[:-math.floor(len(self) * kwargs.get('train_test_split'))]
        elif mode == 'test':
            self.paths_classes = self.paths_classes[-math.floor(len(self) * kwargs.get('train_test_split')):]
        self.image_loader = ImageLoader()
        self.image_loader_transforms = kwargs.get('imageloader_transforms')
        self.preprocessor = Preprocessor()
        self.preprocessor_transforms = kwargs.get('preprocessor_transforms')
        self.image2graph = ImageToGraph()
        self.image2graph_transforms = kwargs.get('image2graph_transforms')
        self.patch_dim = kwargs.get('patch_dim') if kwargs.get('train') else None
        self.batch_size = batch_size
        self.items = 0

    def __len__(self):
        """
        Length of the dataset
        :return: (int)
            length
        """
        return len(self.paths_classes)

    def __getitem__(self, path):
        """
        Returns torch_geometric.Data corresponding to image.
        :param path: (image)
        :return: (torch_geometric.Data, list(torch_geometric.Data))
        """
        item_load = self.image_loader(self.paths_classes[path], **self.image_loader_transforms)
        new_item = (item_load[0], item_load[1],
                    self.preprocessor([item_load[2]], **self.preprocessor_transforms)[0],
                    self.preprocessor.resize(
                        item_load[3],
                        self.preprocessor_transforms.get('resize_dim'),
                        cv2.INTER_NEAREST_EXACT) if self.preprocessor_transforms.get('resize') else item_load[3])
        if self.patch_dim is None:
            graph = self.image2graph((*new_item, None), **self.image2graph_transforms)
        else:
            imgs = extract_patches_2d(new_item[2], patch_size=(self.patch_dim, self.patch_dim),
                                      max_patches=self.batch_size, random_state=self.items)
            masks = extract_patches_2d(new_item[3], patch_size=(self.patch_dim, self.patch_dim),
                                       max_patches=self.batch_size, random_state=self.items)
            graph = [self.image2graph((new_item[0], new_item[1],
                                       imgs[idx],
                                       masks[idx],
                                       ),
                                      **self.image2graph_transforms) for idx in range(self.batch_size)]
            self.items += 1

        return graph


class ImageDataset(Dataset, BaseClass):
    """

    """
    def __init__(self, root_dir, mode, batch_size, **kwargs):
        """
        Class constructor
        :param root_dir: directory with files.
        :param kwargs: DataPathLoaderCall Class
        """
        super().__init__()
        kwargs = self._default_config(ImageDatasetInit, **kwargs)
        # if mode == 'inference':
        #     kwargs.update({'train': False})
        self.root_dir = root_dir
        dp = DataPathLoader(dir_path=root_dir)
        self.paths_classes = dp(**kwargs.get('datapathloader_transforms'))
        random.Random(kwargs.get('seed')).shuffle(self.paths_classes)
        if mode == 'train':
            self.paths_classes = self.paths_classes[:-math.floor(len(self) * kwargs.get('train_test_split'))]
        elif mode == 'test':
            self.paths_classes = self.paths_classes[-math.floor(len(self) * kwargs.get('train_test_split')):]
        self.image_loader = ImageLoader()
        self.image_loader_transforms = kwargs.get('imageloader_transforms')
        self.preprocessor = Preprocessor()
        self.preprocessor_transforms = kwargs.get('preprocessor_transforms')
        self.crop_dim = kwargs.get('crop_dim')
        self.batch_size = batch_size

    def __len__(self):
        """
        Length of the dataset
        :return: (int)
            length
        """
        return len(self.paths_classes)

    def __getitem__(self, path):
        """
        Returns torch_geometric.Data corresponding to image.
        :param path: (image)
        :return: (torch_geometric.Data, list(torch_geometric.Data))
        """
        item_load = self.image_loader(self.paths_classes[path], **self.image_loader_transforms)
        new_item = {'name': item_load[0],
                    'class': item_load[1][0],
                    'image': self.preprocessor([item_load[2]], **self.preprocessor_transforms)[0],
                    'mask': self.preprocessor.resize(
                        item_load[3],
                        self.preprocessor_transforms.get('resize_dim'),
                        cv2.INTER_NEAREST_EXACT) if self.preprocessor_transforms.get('resize') else item_load[3]}
        if self.crop_dim is not None and any(s < self.crop_dim[0] for s in new_item['image'].shape):
            new_item.update({'image': self.preprocessor.resize(new_item['image'], self.crop_dim,
                                                               cv2.INTER_AREA),
                             'mask': self.preprocessor.resize(new_item['mask'], self.crop_dim,
                                                              cv2.INTER_NEAREST_EXACT)})
        im, mask = self.transform(new_item['image'], new_item['mask'], self.crop_dim)
        new_item.update({'image': im,
                         'mask': mask})
        return new_item

    def transform(self, image, mask, output_size):
        """

        :param image:
        :param mask:
        :param output_size: (tup)
        :return:
        """
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        # Random crop
        if self.crop_dim is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=output_size)
            image = tf.crop(image, i, j, h, w)
            mask = tf.crop(mask, i, j, h, w)
        return image, mask


class GraphDataLoader(DataLoader, BaseClass):
    """
    DataLoader for loading graphs
    """
    def __init__(self, root_dir, mode, **kwargs):
        kwargs = self._default_config(GraphDataLoaderInit, **kwargs)
        batch_size = kwargs.get('batch_size')
        self.dataset = GraphDataset(root_dir, mode, batch_size=1,  # batch_size if mode == 'train' else 1,
                                    **kwargs['graph_dataset_kwargs'])
        kwargs.pop('collate_fn', None)
        self.collator = ListCollater(self.dataset)
        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,  # 1 if mode == 'train' else batch_size,
                         shuffle=True if mode == 'train' else False,
                         collate_fn=self.collator.collate_fn
                         )


class ImageDataLoader(DataLoader, BaseClass):
    """
    DataLoader for loading graphs
    """
    def __init__(self, root_dir, mode, **kwargs):
        kwargs = self._default_config(ImageDataLoaderInit, **kwargs)
        batch_size = kwargs.get('batch_size')
        num_workers = kwargs.get('num_workers')
        pin_memory = kwargs.get('pin_memory')
        persistent_workers = kwargs.get('persistent_workers')
        self.dataset = ImageDataset(root_dir, mode, batch_size=batch_size,
                                    **kwargs['image_dataset_kwargs'])
        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=True if mode == 'train' else False,
                         drop_last=True if mode == 'train' else False,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         persistent_workers=persistent_workers
                         )
