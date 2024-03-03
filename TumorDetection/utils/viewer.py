from typing import Optional
import numpy as np
import cv2

from TumorDetection.utils.dict_classes import ViewerClsParams


class Viewer:
    """
    cv2 Visualizer
    """
    params = ViewerClsParams.to_dict()

    @classmethod
    def show_image(cls, image: np.ndarray, win_title: Optional[str] = None):
        """
        Function to show an image
        :param image: np array to show
        :param win_title: (str)
            Window title
        :return:
        """
        if win_title is None:
            win_title = cls.params.get('win_title')
        try:
            cv2.imshow(win_title, image)
            cv2.waitKey(0)
            cv2.destroyWindow(win_title)
        except cv2.error:
            pass

    @classmethod
    def show_masked_image(cls, image: np.ndarray, mask: np.ndarray, win_title: Optional[str], **kwargs):
        """
        Function to show a set of images
        :param image: np array to show
        :param mask: mask to apply
        :param win_title: Window title
        :keyword alpha_weight: (float)
            Alpha weight for combining image and mask
        :keyword beta_weight: (float)
            Beta weight for combining image and mask
        :keyword gamma_weight: (float)
            Gamma weight for combining image and mask
        """
        assert all([s1 == s2 for s1, s2 in zip(image.shape[:2], mask.shape[:2])]),\
            (f'Height and width of images and shapes must be equal.'
             f' Got image shape {image.shape} and mask shape {mask.shape}')
        image = (
            cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB) if (len(image.shape) == 2 or image.shape[-1] != 3) else image
        )
        mask_overlay = cv2.addWeighted(src1=cv2.applyColorMap(mask, cls.params.get('mask_colormap')),
                                       alpha=kwargs.get('alpha_weight', cls.params.get('mask_alpha_weight')),
                                       src2=image,
                                       beta=kwargs.get('beta_weight', cls.params.get('mask_beta_weight')),
                                       gamma=kwargs.get('gamma_weight', cls.params.get('mask_gamma_weight')))

        roi_img = cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask)
        roi_img[mask[:] == 0, ...] = image[mask[:] == 0, ...]

        cls.show_image(roi_img, win_title)
