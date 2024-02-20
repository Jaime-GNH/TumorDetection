import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
import torchvision.transforms.functional as tf

from TumorDetection.utils.dict_classes import TorchDatasetInit, ClassValues
from TumorDetection.utils.base import BaseClass


class TorchDataset(Dataset, BaseClass):
    """
    Torch Image Dataset from DataPathLoader
    """
    def __init__(self, paths, **kwargs):
        """

        :param paths: DataPathLoader __call__ output
        :param kwargs:
        :return:
        """

        self.kwargs = self._default_config(TorchDatasetInit, **kwargs)
        self.imgs_paths = [x[0] for x in paths]
        self.masks_paths = [x[1] for x in paths]
        self.classes = [x[2] for x in paths]
        self.class_values = ClassValues.to_dict()

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.imgs_paths[item], cv2.IMREAD_GRAYSCALE)
        mask = np.max(np.dstack(np.array(
            [
                (cv2.imread(path, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
                for path, clas_ in zip(self.masks_paths[item], self.classes[item])
            ]
        )), axis=-1)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)

        image = tt.Resize(self.kwargs.get('resize_dim'),
                          antialias=None)(image)
        mask = tt.Resize(self.kwargs.get('resize_dim'),
                         antialias=None,
                         interpolation=tt.InterpolationMode.NEAREST_EXACT)(mask)

        if random.random() > 0.5:
            i, j, h, w = tt.RandomCrop.get_params(
                image, output_size=self.kwargs.get('output_dim'))
            image = tf.crop(image, i, j, h, w)
            mask = tf.crop(mask, i, j, h, w)
        else:
            image = tt.Resize(self.kwargs.get('output_dim'),
                              antialias=None)(image)
            mask = tt.Resize(self.kwargs.get('output_dim'),
                             antialias=None,
                             interpolation=tt.InterpolationMode.NEAREST_EXACT)(mask)

        if self.kwargs.get('rotatation_degrees') is not None:
            angle = tt.RandomRotation.get_params(self.kwargs.get('rotatation_degrees'))
            image = tf.rotate(image, angle, tt.InterpolationMode.BILINEAR)
            mask = tf.rotate(mask, angle, tt.InterpolationMode.NEAREST_EXACT)

        if self.kwargs.get('max_brightness') is not None:
            brightness = random.random()*self.kwargs.get('max_brightness')
            image = tf.adjust_brightness(image, brightness_factor=brightness)
            mask = tf.adjust_brightness(mask, brightness_factor=brightness)

        if self.kwargs.get('max_contrast') is not None:
            contrast = random.random() * self.kwargs.get('max_contrast')
            image = tf.adjust_contrast(image, contrast_factor=contrast)
            mask = tf.adjust_contrast(mask, contrast_factor=contrast)

        if self.kwargs.get('max_saturation') is not None:
            saturation = random.random() * self.kwargs.get('max_saturation')
            image = tf.adjust_saturation(image, saturation_factor=saturation)
            mask = tf.adjust_saturation(mask, saturation_factor=saturation)

        # Random horizontal flipping
        if self.kwargs.get('horizontal_flip_prob') is not None:
            if random.random() > self.kwargs.get('horizontal_flip_prob'):
                image = tf.hflip(image)
                mask = tf.hflip(mask)

        # Random vertical flipping
        if self.kwargs.get('vertical_flip_prob') is not None:
            if random.random() > self.kwargs.get('vertical_flip_prob'):
                image = tf.vflip(image)
                mask = tf.vflip(mask)

        label = torch.tensor(self.class_values[self.classes[item][0]]) if mask.sum() > 0 else torch.tensor(0)

        return (image,
                mask,
                label)
