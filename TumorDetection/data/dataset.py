from typing import List, Optional, Any
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as tfn
import torchvision.transforms as tt
import torchvision.transforms.functional as tf

from TumorDetection.utils.dict_classes import TorchDatasetInit, ClassValues
from TumorDetection.utils.base import BaseClass


class TorchDataset(Dataset, BaseClass):
    """
    Torch Image Dataset from DataPathLoader
    """
    def __init__(self, paths: List[str, List[Optional[str]], List[Optional[int]]],
                 **kwargs):
        """
        Torch Dataset class constructor.
        :param paths: DataPathLoader __call__ output
        :keyword resize_dim: (Tuple[int,int])
            Image resize dimension.
        :keyword output_dim: (Tuple[int,int])
            Image output (model input) dimension.
        :keyword rotation_degrees: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]]
            Rotation degrees margins.
        :keyword range_brightness: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]]
            Max brightness adjustement possible or range in min-max brightness
        :keyword range_saturation: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]]
            Max saturation adjustment possible or range in min-max saturation.
        :keyword range_contrast: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]]
            Max contrast adjustment possible or range in min-max contrast.
        :keyword horizontal_flip_prob: Optional[float]
            Horizontal flip over image probability.
        :keyword vertical_flip_prob: Optional[float]
            Vertical flip over image probability.
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

        if angle := self.kwargs.get('rotatation_degrees') is not None:
            angle = self._get_random_param(angle, 'rotation_degrees')
            image = tf.rotate(image, angle, tt.InterpolationMode.BILINEAR)
            mask = tf.rotate(mask, angle, tt.InterpolationMode.NEAREST_EXACT)

        if brightness := self.kwargs.get('range_brightness') is not None:
            brightness = self._get_random_param(brightness, 'range_brightness')
            image = tf.adjust_brightness(image, brightness_factor=brightness)
            mask = tf.adjust_brightness(mask, brightness_factor=brightness)

        if contrast := self.kwargs.get('range_contrast') is not None:
            contrast = self._get_random_param(contrast, 'max_contrast')
            image = tf.adjust_contrast(image, contrast_factor=contrast)
            mask = tf.adjust_contrast(mask, contrast_factor=contrast)

        if saturation := self.kwargs.get('range_saturation') is not None:
            saturation = self._get_random_param(saturation, 'range_saturation')
            image = tf.adjust_saturation(image, saturation_factor=saturation)
            mask = tf.adjust_saturation(mask, saturation_factor=saturation)

        # Random horizontal flipping
        if hflip_prob := self.kwargs.get('horizontal_flip_prob') is not None:
            if random.random() > hflip_prob:
                image = tf.hflip(image)
                mask = tf.hflip(mask)

        # Random vertical flipping
        if vflip_prob := self.kwargs.get('vertical_flip_prob') is not None:
            if random.random() > vflip_prob:
                image = tf.vflip(image)
                mask = tf.vflip(mask)

        mask = tfn.one_hot(mask.to(torch.long).squeeze(),
                           num_classes=2).double().permute(2, 1, 0)
        label = (torch.tensor(self.class_values[self.classes[item][0]])
                 if mask.sum() > 0 else torch.tensor(0))

        return (image,
                mask,
                label)

    @staticmethod
    def _get_random_param(param: Any, name: Optional[str] = None) -> float:
        """
        Check if specified param in __getitem__ is well typed and return the randomized parameter to use.
        :param param: Parameter to be checked
        :param name: Parameter name
        :return: parameter to use.
        """
        if isinstance(param, int):
            return random.random() * param
        elif isinstance(param, tuple):
            return param[0] + random.random() * (param[1] - param[0])
        else:
            raise TypeError(f'param {name} bust be None, int or tuple. Got {type(param)}')
