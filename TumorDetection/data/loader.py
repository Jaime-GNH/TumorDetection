from typing import Optional, List, Union
import glob


from TumorDetection.utils.dict_classes import DataPathLoaderCall
from TumorDetection.utils.base import BaseClass


class DataPathLoader(BaseClass):
    """
    Class for definind the path for images, masks and class in BUSI Dataset.
    Dataset BUSI available on: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
    """
    def __init__(self, dir_path: str, substring_filter: Optional[str] = None):
        self.dir_path = dir_path
        if substring_filter is None:
            self.imgs_paths = glob.glob(dir_path + r'\*\*).png')
        else:
            self.imgs_paths = [x for x in glob.glob(dir_path + r'\*\*).png') if substring_filter not in x]

    def __call__(self, **kwargs) -> List[Union[str, List[Optional[str]]]]:
        """
        Returns the image, masks, class pairings in BUSI Dataset.
        :keyword find_mask: (bool, True)
            Whether to find associated mask or not
        :keyword map_classes: (Optional[bool])
            A dicionary mapping each class value to an integer.
        :keyword pair_masks: (bool, True)
            If consider multiple mask of a single image as the same mask or not.
        :return: List of (images paths, list of possible masks, classes for each mask)
        """
        kwargs = self._default_config(DataPathLoaderCall, **kwargs)
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

    def _find_masks(self) -> List[List[str]]:
        """
        Find masks associated to image_paths
        :return: list with len equal to self.image_path with associated mask(s) path.
        """
        return [glob.glob(img_path.split('.')[0]+'_mask*') for img_path in self.imgs_paths]

    def _map_classes(self, join_classes: dict) -> List[List[str]]:
        """
        Convert a tuple of clases into a single one.
        :param join_classes: dict with al class map as {'benign': 'tumor', 'malignant': 'tumor', 'normal': 'normal'}
        :return: New classes
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
