from RL_TD.Utils.DictClasses import DataPathsCall, ImageLoaderCall
from RL_TD.Utils.BaseClass import BaseClass
from RL_TD.Utils.Utils import apply_function2list

import glob
import cv2
import numpy as np


class DataPaths(BaseClass):
    """
    Class for definind the path for images, masks and class in BUSI Dataset.
    Dataset BUSI available on: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.imgs_paths = glob.glob(dir_path + r'\*\*).png')

    def __call__(self, **kwargs):
        """

        :param kwargs:
        :return: list(str, list(str), (int))
            List of (images paths, list of possible masks, classes for each mask)
        """
        kwargs = self._default_config(DataPathsCall, **kwargs)
        if kwargs.get('find_masks', True):
            self.masks_paths = self._find_masks()
            self.classes = [[path.split('\\')[-2] for path in mask_paths] for mask_paths in self.masks_paths]
            map_classes = kwargs.get('map_classes')
            pair_masks = kwargs.get('pair_masks', True)

            if map_classes is not None:
                self.classes = self._map_classes(map_classes)
            if not pair_masks:
                self._unpair_masks()

        else:
            self.masks_paths = [None]*len(self.imgs_paths)
            self.classes = [None]*len(self.imgs_paths)
        return list(
            zip(
                self.imgs_paths,
                self.masks_paths,
                self.classes
            )
        )

    def _find_masks(self):
        """
        Find masks associated to image_paths
        :return: list(list(str))
            list with len equal to self.image_path
        """
        return [glob.glob(img_path.split('.')[0]+'_mask*') for img_path in self.imgs_paths]

    def _map_classes(self, join_classes):
        """
        Convert a tuple of clases into a single one.
        :param join_classes: (dict)
            dict with al class map as {'benign': 'tumor', 'malignant': 'tumor', 'normal': 'normal'}
        :return: (list(list(str)))
            New classes
        """
        return [[*map(join_classes.get, lst)] for lst in self.classes]

    def _unpair_masks(self):
        """
        Duplicates the image_paths that hav one or more mask associated.
        """
        imgs_paths, masks_paths, classes = [], [], []
        for img_path, masks_path, classes_ in zip(self.imgs_paths, self.masks_paths, self.classes):
            for mask_path, clas_ in zip(masks_path, classes_):
                imgs_paths += img_path
                masks_paths += [mask_path]
                classes += clas_
        self.imgs_paths = imgs_paths
        self.masks_paths = masks_paths
        self.classes = classes


class ImageLoader(BaseClass):
    """
    Class for image loading
    """
    def __call__(self, paths_classes, **kwargs):
        """
        Main function for readin images.
        :param path_classes: list(tuple(str, list(str), list(str)))
            list(imgs_paths, [masks_paths], [associated_classes]). Result of DataPaths
        :keyword read_mode: (str, 'gray')
            Reading mode for images
        :return: list(tuple(list(str), list(list(str)), list(cv2.Image), list(cv2.Image)))
            tuple(images_paths, [classes], np.array(image), np.array(mask))
            A tuple containing images_paths, associated_classes, images and masks.
        """
        # paths_classes = path_classes
        kwargs = self._default_config(ImageLoaderCall, **kwargs)
        return apply_function2list(paths_classes,
                                   self._process_tuple,
                                   **kwargs)
        # return list(map(lambda tup: self._process_tuple(tup,
        #                                                 kwargs.get('read_mode'),
        #                                                 kwargs.get('class_values')),
        #                 paths_classes))

    def _process_tuple(self, tup, read_mode, class_values):
        """
        :param read_mode: (str)
        :return: tuple(list(str), list(list(str)), list(cv2.Image), list(cv2.Image))
        """
        return tup[0], tup[-1], self._read_image(tup[0], read_mode), self._read_mask(tup[1], tup[2], class_values)

    @staticmethod
    def _read_image(path, read_mode):
        """

        :param path: (str)
            path to image
        :param read_mode: (str)
            reading mode
        :return: (np.ndarray)
            image
        """
        return cv2.imread(path, ReadingModes.get(read_mode))

    def _read_mask(self, paths, classes, class_values):
        """

        :param paths: (str)
            paths to masks
        :param classes: (str)
            classes
        :return: (np.ndarray)
            mask
        """
        return np.max(np.dstack(np.array(
            [
                class_values[clas_]*(self._read_image(path, 'gray')/255.).astype(np.uint8)
                for path, clas_ in zip(paths, classes)
            ]
        )), axis=-1)


if __name__ == '__main__':
    import os
    import random
    from RL_TD.Utils.DictClasses import DataPath, ReadingModes
    from RL_TD.Utils.DictClasses import BaseClassMap, MappedClassValues
    from RL_TD.Utils.Viewer import Viewer
    from RL_TD.Data.Preprocess import Preprocessor

    Dp = DataPaths(dir_path=DataPath.get('dir_path'))
    paths_classes = Dp(map_classes=BaseClassMap.to_dict())

    result = ImageLoader()(paths_classes, class_values=MappedClassValues.to_dict())
    images = Preprocessor()([r[2] for r in result])

    idx = random.choice(range(len(result)))
    Viewer.show_masked_image(result[idx][2], result[idx][3],
                             win_title=f'Image: {os.path.splitext(os.path.basename(result[idx][0]))[0]}.'
                                       f' Class: {result[idx][1][0]}')


