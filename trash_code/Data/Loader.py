
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
        if isinstance(paths_classes, tuple) and not isinstance(paths_classes[0], tuple):
            return self._process_tuple(paths_classes, **kwargs)
        elif isinstance(paths_classes, list) and isinstance(paths_classes[0], tuple):
            return apply_function2list(paths_classes,
                                       self._process_tuple,
                                       **kwargs)
        else:
            raise ValueError(f'paths_classes must be list of tuples or single tuple. Got {type(paths_classes)}')

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
