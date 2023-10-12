import os


class ConfigClass:

    @classmethod
    def to_dict(cls):
        """

        :return:
        """
        return {k: v for k, v in vars(cls).items() if not k.startswith('__')}


class Classes(ConfigClass):
    normal = 0
    benign = 1
    malignant = 2


class LoaderInit(ConfigClass):
    imgs_regex = r'\*\*).png'
    read_mode = 'gray'  # [color, gray]


class LoaderCall(ConfigClass):
    classes = Classes.to_dict()
    join_classes = None  # {['bening','malignant']: 'tumor'}
    add_masks = True


class LoaderConfig(ConfigClass):
    dir_path = [
            os.path.join(dir_path, 'Dataset_BUSI_with_GT')
            for dir_path, dir_name, _ in os.walk(os.getcwd())
            if 'Dataset_BUSI_with_GT' in dir_name
        ][0]
    init = LoaderInit.to_dict()
    call = LoaderCall.to_dict()



