from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import PreprocessorCall
from TumorDetection.Utils.Utils import apply_function2list

import numpy as np
import cv2


class Preprocessor(BaseClass):
    """
    Class with methods for preprocessing images.
    """

    def __call__(self, images, **kwargs):
        """
        Main function.
        :param images: list(np.ndarray)
            Set of image to preprocess
        :keyword adjust_exposure: (bool)
            Adjust or not the exposure (brightness and contrast)
        :keyword clip_hist_percent: (int)
            Histogram clipping percent. Only used if adjust_exposure is True.
        :return: (dict(list))
            Dictionary with applied transforms
        """
        kwargs = self._default_config(PreprocessorCall, **kwargs)
        result = {'original': images}
        if kwargs.get('resize'):
            result.update({
                'original': apply_function2list(
                    result['original'],
                    Preprocessor.resize,
                    dim=kwargs['resize_dim'],
                    interpolation=kwargs['interpolation_method']
                )
            })
        # 1. Scale inversion
        if kwargs.get('invert_grayscale'):
            result.update({
                'inverted': apply_function2list(
                    result['original'],
                    Preprocessor.invert_grayscale)
            })

        # 2. Exposure correction
        if kwargs.get('adjust_exposure'):
            result.update({
                'adjusted': apply_function2list(
                    result[list(result)[-1]],
                    Preprocessor.automatic_brightness_and_contrast,
                    clip_hist_percent=kwargs.get('clip_hist_percent'))
            })

        # 3. Contrast enhancing
        if kwargs.get('apply_clahe'):
            result.update({
                'clahe': apply_function2list(
                    result[list(result)[-1] if kwargs.get('clahe_over_last') else list(result)[-2]],
                    Preprocessor.apply_clahe,
                    clip_limit=kwargs.get('clip_limit'))
            })

        # 4. Masks aproximation
        if kwargs.get('apply_threshold'):
            for threshold_std in kwargs.get('img_thresholds_std'):
                result.update({
                    f'threshold_{threshold_std:.2f}': apply_function2list(
                        result[[k for k in list(result) if not k.startswith(('threshold', 'contour'))][-1]],
                        Preprocessor.apply_threshold,
                        threshold_std=((-1) ** (1 ^ kwargs.get('invert_grayscale'))) * threshold_std)
                })

                if kwargs.get('detect_contours'):
                    result.update({
                        f'contour_{threshold_std:.2f}': apply_function2list(
                            result[list(result)[-1]],
                            Preprocessor.detect_countours)
                    })
        if kwargs.get('mode') == 'stacked':
            images = apply_function2list(
                list(map(list, zip(*result.values()))),
                np.dstack)
        elif kwargs.get('mode') == 'last':
            images = result.get(list(result)[-1])
        else:
            raise ValueError(f'param mode: {kwargs.get("mode")} not contemplated')
        return images

    @classmethod
    def resize(cls, img, dim, interpolation):
        """
        resizes image
        :param img: (np.ndarray)
            Image to be resized
        :param dim: (tuple)
            Dimension to get
        :param interpolation: cv2.INTERPOLATION
            How to interpolate pixels
        :return: ()
            New image
        """
        return cv2.resize(img, dim, interpolation=interpolation)

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
        return auto_result

    @classmethod
    def apply_threshold(cls, img, threshold_std=1):
        """
        Given a threshold param any value over is set to 255 and any below set to 0
        :param img: (np.ndarray)
            image
        :param threshold_std: (float, None)
            threshold = mean(img) + threshold_std*std(img)
        :return: (np.ndarray)
            new_image
        """
        threshold = np.mean(img) + threshold_std * np.std(img)
        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return thresh

    @classmethod
    def detect_countours(cls, thresh):
        """
        Given a thresholded images extracts the contours
        :param thresh: (np.ndarray)
            Thresholded image
        :return: (np.ndarray)
            Contoured image
        """
        return cv2.dilate(src=cv2.Canny(thresh, 0, 255),
                          kernel=np.ones((1, 1), np.uint8))

    @classmethod
    def apply_clahe(cls, img, clip_limit):
        """

        :param img
        :param clip_limit:
        :return:
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit)
        return clahe.apply(img)
