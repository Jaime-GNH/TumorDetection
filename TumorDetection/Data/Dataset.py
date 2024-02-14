import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split
import torchvision.transforms as tt
import torchvision.transforms.functional as tf

from TumorDetection.Utils.DictClasses import PatchDatasetCall, TorchDatasetInit, ClassValues
from TumorDetection.Utils.BaseClass import BaseClass


class PatchDataset(BaseClass):
    """
    Makes a Batched dataset from a set of images and masks
    """
    def __call__(self, images, masks, **kwargs):
        """

        :param images: List(np.array)
        :param masks: List(np.array)
        :param kwargs:
        :return:
        """

        kwargs = self._default_config(PatchDatasetCall, **kwargs)
        assert (kwargs.get('resize_dim')[0] - kwargs.get('patch_dim')[0]) % kwargs.get('patch_step') == 0, \
            f'(Resize dim - patch dim) % patch_step must be 0.'
        assert (kwargs.get('resize_dim')[1] - kwargs.get('patch_dim')[1]) % kwargs.get('patch_step') == 0, \
            f'(Resize dim - patch dim) % patch_step must be 0.'

        if kwargs.get('resize'):
            images = [cv2.resize(image, kwargs.get('resize_dim'),
                                 interpolation=kwargs.get('interpolation_method')) for image in images]
            masks = [cv2.resize(mask, kwargs.get('resize_dim'),
                                interpolation=cv2.INTER_NEAREST_EXACT) for mask in masks]
        # num_classes = np.max(np.array(masks))
        if kwargs.get('normalize'):
            images = [image.astype('float') / 255. for image in images]

        im_tr, im_ts, mask_tr, mask_ts = train_test_split(np.stack(images), np.stack(masks),
                                                          test_size=kwargs.get('test_size'),
                                                          shuffle=kwargs.get('shuffle'),
                                                          random_state=kwargs.get('random_state'))
        if kwargs.get('mode') == 'images':
            return im_tr, im_ts, mask_tr.squeeze(), mask_ts.squeeze()
            # return (im_tr, im_ts,
            #         keras.utils.to_categorical(mask_tr, num_classes=num_classes),
            #         keras.utils.to_categorical(mask_ts, num_classes=num_classes))
        elif kwargs.get('mode') == 'patches':
            x_tr = np.concatenate([self.make_patches(image,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for image in im_tr])
            y_tr = np.concatenate([self.make_patches(mask,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for mask in mask_tr])

            x_ts = np.concatenate([self.make_patches(image,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for image in im_ts])
            y_ts = np.concatenate([self.make_patches(mask,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for mask in mask_ts])
            if kwargs.get('filter_empty'):
                pass
                idx = np.where(np.max(y_tr, axis=(1, 2)).flatten() != 0)
                x_tr = x_tr[idx]
                y_tr = y_tr[idx]

            return x_tr, x_ts, y_tr.squeeze(), y_ts.squeeze()
            # return (x_tr, x_ts,
            #         keras.utils.to_categorical(y_tr, num_classes=num_classes+1),
            #         keras.utils.to_categorical(y_ts, num_classes=num_classes+1))
        else:
            raise ValueError(f'mode must be one of: "images", "patches". Got {kwargs.get("mode")}')

    @staticmethod
    def make_patches(image, patch_dim, patch_step):
        """
        (HxW) -> (PHxPWxHxW)
        :return:
        """
        return patchify(image,
                        patch_dim,
                        patch_step)

    @staticmethod
    def unpatch_image(patches, image_dim):
        """
        (PHxPWxHxW) -> (HxW)
        :param patches:
        :param image_dim:
        :return:
        """
        return unpatchify(patches, image_dim)


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

        i, j, h, w = tt.RandomCrop.get_params(
            image, output_size=self.kwargs.get('crop_dim'))
        image = tf.crop(image, i, j, h, w)
        mask = tf.crop(mask, i, j, h, w)

        if self.kwargs.get('rotatation_degrees'):
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

        return (image.reshape(*image.shape, 1),
                mask.reshape(*mask.shape, 1),
                label)
