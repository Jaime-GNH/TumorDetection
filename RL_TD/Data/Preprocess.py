from RL_TD.Utils.BaseClass import BaseClass
from RL_TD.Utils.DictClasses import PreprocessorCall
from RL_TD.Utils.Utils import apply_function2list


import numpy as np
import cv2


class Preprocessor(BaseClass):
    """
    Class with methods for preprocessing images.
    """
    def __init__(self):
        """
        Class Constructor
        """
        pass

    def __call__(self, images, **kwargs):
        """
        Main function.
        :param images: list(np.ndarray)
            Set of image to preprocess
        :keyword adjust_exposure: (bool)
            Adjust or not the exposure (brightness and contrast)
        :keyword clip_hist_percent: (int)
            Histogram clipping percent. Only used if adjust_exposure is True.
        :return:
        """
        kwargs = self._default_config(PreprocessorCall, **kwargs)

        # 1. Scale inversion
        if kwargs.get('invert_grayscale'):
            images = apply_function2list(images,
                                         Preprocessor.invert_grayscale)

        # 2. Exposure correction
        if kwargs.get('adjust_exposure'):
            images, _, _ = apply_function2list(
                images,
                Preprocessor.automatic_brightness_and_contrast,
                clip_hist_percent=kwargs.get('clip_hist_percent')
            )

        return images

    @classmethod
    def invert_grayscale(cls, img):
        """
        Inverts grayscale
        :param img: (np.ndarray)
            Image to be scale-flipped
        :return: (nd.array)
            New image
        """
        assert len(img.shape) == 2, f'img must be grayscale. Got an image of dimensions {img.shape}'
        return cv2.bitwise_not(img)

    @classmethod
    def convert_scale(cls, img, alpha, beta):
        """Add bias and gain to an image with saturation arithmetics. Unlike
        cv2.convertScaleAbs, it does not take an absolute value, which would lead to
        nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
        becomes 78 with OpenCV, when in fact it should become 0).
        :param img: (np.array)
            Image to convert scale
        :param alpha: (int)
            Brightness
        :param beta: (int)
            Contrast
        """

        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    @classmethod
    def automatic_brightness_and_contrast(cls, img, clip_hist_percent):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
        # Calculate cumulative distribution from the histogram
        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cls.convert_scale(img, alpha=alpha, beta=beta)
        return auto_result, alpha, beta
