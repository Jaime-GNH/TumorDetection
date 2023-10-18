import os
import cv2


class DictClass:

    @classmethod
    def to_dict(cls):
        """

        :return:
        """
        return {k: v for k, v in vars(cls).items() if not k.startswith('__')}

    @classmethod
    def get(cls, name):
        """

        :return:
        """
        return vars(cls)[name]


# [MAPS]
class ClassValues(DictClass):
    normal = 0
    benign = 1
    malignant = 2


class MappedClassValues(DictClass):
    normal = 0
    tumor = 1


class BaseClassMap(DictClass):
    normal = 'normal'
    benign = 'tumor'
    malignant = 'tumor'


# [PARAMS]
class DataPathDir(DictClass):
    dir_path = [
        os.path.join(dir_path, 'Dataset_BUSI_with_GT')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'Dataset_BUSI_with_GT' in dir_name
    ][0]


class ViewerClsParams(DictClass):
    win_title = 'Viewer'
    mask_colormap = cv2.COLORMAP_RAINBOW
    mask_alpha_weight = 0.3
    mask_beta_weight = 0.7
    mask_gamma_weight = 0


class ReadingModes(DictClass):
    gray = cv2.IMREAD_GRAYSCALE
    color = cv2.IMREAD_COLOR
    unchanged = cv2.IMREAD_UNCHANGED


# [DEFAULT CONFIGS]
class DataPathLoaderCall(DictClass):
    find_masks = True
    map_classes = None  # {'bening': 'tumor','malignant': 'tumor', 'normal': normal}
    pair_masks = True


class ImageLoaderCall(DictClass):
    read_mode = 'gray'
    class_values = ClassValues.to_dict()


class PreprocessorCall(DictClass):
    invert_grayscale = True
    adjust_exposure = True
    apply_clahe = True
    apply_threshold = False
    detect_contours = False
    clahe_over_last = True
    img_thresholds_std = [0, 1/4, 1/2, 1]
    clip_hist_percent = 25
    clip_limit = 5.0

