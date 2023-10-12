import cv2


class ParamClass:

    @classmethod
    def get(cls, name):
        """

        :return:
        """
        return vars(cls)[name]


class ReadingModes(ParamClass):
    gray = cv2.IMREAD_GRAYSCALE
    color = cv2.IMREAD_COLOR
    unchanged = cv2.IMREAD_UNCHANGED
