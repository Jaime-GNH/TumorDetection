from TumorDetection.utils.dict_classes import DictClass


class BaseClass:
    @staticmethod
    def _default_config(default_config: DictClass, **kwargs) -> dict:
        """
        Updates Default config with external kwargs
        :param default_config: DefaultConfig class
            Default config definition
        :param kwargs: (dict)
            kwargs
        :return: (dict)
            Updated kwargs
        """
        assert isinstance(default_config, DictClass),\
            f'default_config must be a DictClass class object.'
        default_kwargs = default_config.to_dict()
        if kwargs is not None:
            default_kwargs.update(kwargs)
        return default_kwargs
