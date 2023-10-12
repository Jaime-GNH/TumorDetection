from RL_TD.config import LoaderConfig
from RL_TD.params import ReadingModes

import glob
import cv2


class DataPaths:
    """
    Class for definind the path for images, masks and class in BUSI Dataset.
    Dataset BUSI available on: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
    """
    def __init__(self, dir_path, **kwargs):
        self.dir_path = dir_path
        self.imgs_paths = glob.glob(dir_path + r'\*\*).png')

    def __call__(self, **kwargs):
        """

        :param kwargs:
        :return: list(str, list(str), (int))
            List of (images paths, list of possible masks, classes for each mask)
        """
        if kwargs.get('find_mask', True):
            self.masks_paths = self._find_masks()
            self.classes = [[path.split('\\')[-2] for path in mask_paths] for mask_paths in self.masks_paths]
            map_classes = kwargs.get('map_classes')
            pair_masks = kwargs.get('pair_masks', True)

            if map_classes is not None:
                self._map_classes(join_classes)
            if not pair_masks:
                self._unpair_masks()

        else:
            return list(
                zip(self.imgs_paths,
                    [None]*len(self.imgs_paths),
                    [None]*len(self.imgs_paths)
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

        :return:
        """
        raise NotImplementedError('No se ha implementado _unpair_masks')


class ImageLoader:
    """
    Class for image loading
    """
    def __init__(self, imgs_paths, masks_paths,
                 **kwargs):
        """
        Class constructor

        """

        self.imgs_paths = imgs_paths
        self.mask_paths = masks_paths

    def __call__(self, **kwargs):
        """

        :param kwargs:
        :return: tuple(list(cv2.Image), list(cv2.Image))
            A tuple containing images, masks and associated classes.
        """
        ###
        read_mode = kwargs.get('read_mode', 'gray')
        images = self._read_imgs(read_mode)
        if self.mask_paths is not None:
            masks = self._read_imgs('gray')
            return images, masks
        else:
            return images, [None]*len(images)

    def _read_imgs(self, read_mode):
        """

        :param read_mode: (str)
        :return:
        """
        return [cv2.imread(img, ReadingModes.get(read_mode)) for img in self.imgs_paths]

if __name__ == '__main__':

    Dp = DataPath(dir_path=LoaderConfig.dir_path,
                  **LoaderConfig.init)
    images_paths, masks_paths, classes = Dp(**LoaderConfig.call)
