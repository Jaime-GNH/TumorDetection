from RL_TD.config import config

import glob


class ImageLoader:
    """
    Class for image loading
    """
    def __init__(self, dir_path,
                 **kwargs):
        """
        Class constructor
        :param dir_path: (str, abspath to resources/Dataset_BUSI_with_GT
        """
        self.dir_path = dir_path
        self.imgs_paths = glob.glob(dir_path+kwargs.get('imgs_regex' + r'\*\*).png'))

    def __call__(self, **kwargs):
        """

        :param kwargs:
        :return: tuple(list(cv2.Image), list(cv2.Image), list(int))
            A tuple containing images, masks and associated classes.
        """
        ###
        join_classes = kwargs.get('join_classes')
        add_masks = kwargs.get('add_masks', True)
        if join_classes is not None:
            self._join_classes(join_classes)
        if add_masks is not None:
            self._add_masks()

        return images, masks, classes

    def _join_classes(self, join_classes):
        """
        Convert a tuple of clases into a single one.
        :return:
        """
        raise NotImplementedError('No se ha implementado _join_classes')

    def _add_masks(self):
        """

        :return:
        """
        raise NotImplementedError('No se ha implementado _add_masks')


if __name__ == '__main__':

    Il = ImageLoader(dir_path=config['Data']['Loader']['dir_path'],
                     **config['Data']['Loader']['init'])
    images, masks, classes = Il(**config['Data']['Loader']['call'])

