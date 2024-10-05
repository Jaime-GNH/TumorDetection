from typing import List, Optional, Any, Union, Tuple
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
import torchvision.transforms.functional as tf

from TumorDetection.utils.dict_classes import ClassValues
from TumorDetection.utils.base import BaseClass


class TorchDataset(Dataset, BaseClass):
    """
    Torch Image Dataset from DataPathLoader
    """
    def __init__(self, paths: List[Union[str, List[Optional[str]]]],
                 resize_dim: Tuple[int, int] = (512, 512), output_dim: Tuple[int, int] = (256, 256),
                 crop_prob: Optional[float] = 0.5,
                 rotation_degrees: Optional[Union[int, float]] = 180,
                 range_brightness: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]] = None,
                 range_contrast: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]]] = None,
                 horizontal_flip_prob: Optional[float] = 0.25, vertical_flip_prob: Optional[float] = 0.25):
        """
        Torch Dataset class constructor.
        :param paths: DataPathLoader __call__ output
        :param resize_dim: Image resize dimension.
        :param output_dim: Image output (model input) dimension.
        :param crop_prob: Probability for cropping over resizing image.
        :param rotation_degrees: Rotation degrees margins.
        :param range_brightness: Max brightness adjustement possible or range in min-max brightness
        :param range_contrast: Max contrast adjustment possible or range in min-max contrast.
        :param horizontal_flip_prob: Horizontal flip over image probability.
        :param vertical_flip_prob: Vertical flip over image probability.
        """
        self.kwargs = {
            'resize_dim': resize_dim,
            'output_dim': output_dim,
            'crop_prob': crop_prob,
            'rotation_degrees': rotation_degrees,
            'range_brightness': range_brightness,
            'range_contrast': range_contrast,
            'horizontal_flip_prob': horizontal_flip_prob,
            'vertical_flip_prob': vertical_flip_prob
        }
        self.imgs_paths = [x[0] for x in paths]
        self.masks_paths = [x[1] for x in paths]
        self.classes = [x[2] for x in paths]
        self.class_values = ClassValues.to_dict()

    def __len__(self):
        return len(self.imgs_paths)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augments an image
        :param image: given image to augment
        :param mask: associated mask
        :return: new image and mask
        """
        if (crop_prob := self.kwargs.get('crop_prob')) is not None:
            image = tt.Resize(self.kwargs.get('resize_dim'),
                              antialias=None)(image)
            mask = tt.Resize(self.kwargs.get('resize_dim'),
                             antialias=None,
                             interpolation=tt.InterpolationMode.NEAREST_EXACT)(mask)

            if random.random() < crop_prob:
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
        else:
            image = tt.Resize(self.kwargs.get('output_dim'),
                              antialias=None)(image)
            mask = tt.Resize(self.kwargs.get('output_dim'),
                             antialias=None,
                             interpolation=tt.InterpolationMode.NEAREST_EXACT)(mask)

        if (angle := self.kwargs.get('rotation_degrees')) is not None:
            angle = self._get_random_param(angle, 'rotation_degrees')
            image = tf.rotate(image, angle, tt.InterpolationMode.BILINEAR)
            mask = tf.rotate(mask, angle, tt.InterpolationMode.NEAREST)

        if (brightness := self.kwargs.get('range_brightness')) is not None:
            brightness = self._get_random_param(brightness, 'range_brightness')
            image = tf.adjust_brightness(image, brightness_factor=brightness)

        if (contrast := self.kwargs.get('range_contrast')) is not None:
            contrast = self._get_random_param(contrast, 'max_contrast')
            image = tf.adjust_contrast(image, contrast_factor=contrast)

        # Random horizontal flipping
        if (hflip_prob := self.kwargs.get('horizontal_flip_prob')) is not None:
            if random.random() < hflip_prob:
                image = tf.hflip(image)
                mask = tf.hflip(mask)

        # Random vertical flipping
        if (vflip_prob := self.kwargs.get('vertical_flip_prob')) is not None:
            if random.random() < vflip_prob:
                image = tf.vflip(image)
                mask = tf.vflip(mask)

        return image, mask

    @staticmethod
    def _get_random_param(param: Any, name: Optional[str] = None) -> float:
        """
        Check if specified param in __getitem__ is well typed and return the randomized parameter to use.
        :param param: Parameter to be checked
        :param name: Parameter name
        :return: parameter to use.
        """
        if isinstance(param, int):
            return random.uniform(-param, param)
        elif isinstance(param, (tuple, list)):
            return random.uniform(param[0], param[1])
        else:
            raise TypeError(f'param {name} bust be None, int or tuple. Got {type(param)}')

    def __getitem__(self, item):
        image = cv2.imread(self.imgs_paths[item], cv2.IMREAD_GRAYSCALE)
        mask = np.max(np.dstack(np.array(
            [
                (cv2.imread(path, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
                for path, clas_ in zip(self.masks_paths[item], self.classes[item])
            ]
        )), axis=-1)  # (0, 255)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)  # (0, 1)
        mask[torch.where(torch.eq(mask, 1.))] = self.class_values[self.classes[item][0]]
        image, mask = self._augment(image, mask)
        return (image,
                mask.squeeze().to(torch.long))
