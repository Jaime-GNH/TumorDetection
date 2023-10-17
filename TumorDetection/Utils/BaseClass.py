from TumorDetection.Utils.DictClasses import DictClass


class BaseClass:
    @staticmethod
    def _default_config(default_config, **kwargs):
        """
        Updates Default config with external kwargs
        :param default_config: DefaultConfig class
            Default config definition
        :param kwargs: (dict)
            kwargs
        :return: (dict)
            Updated kwargs
        """
        assert issubclass(default_config, DictClass),\
            f'default_config must be a DefaultConfigClass subclass object.'
        default_kwargs = default_config.to_dict()
        default_kwargs.update(kwargs)
        return default_kwargs
