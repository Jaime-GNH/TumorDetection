import os
from typing import Union, Optional

import torch
import lightning.pytorch as pl


def load_ckpt(ckpt_dir: str, model_name: str,
              lightningmodel: Union[pl.LightningModule, type],
              torchmodel: Optional[Union[torch.nn.Module, type]] = None,
              torchmodel_kwargs: Optional[dict] = None) -> pl.LightningModule:
    """
    Load model from .ckpt file
    :param ckpt_dir: Checkpoint directory
    :param model_name: Model name identifier
    :param lightningmodel: Lightning Module model wrapper
    :param torchmodel: Neural Network Architecture
    :param torchmodel_kwargs: Keyword arguments for initializing Neural Network
    :return: Model with trained weights
    """
    assert {'load', '__cls__'}.issubset(
        [f for f in dir(lightningmodel.__cls__()) if callable(getattr(lightningmodel.__cls__(), f))]
    ), f'Lightning Model {lightningmodel.__name__} must have `load` and `__cls__` methods implemented'
    if not isinstance(lightningmodel, type):
        model = lightningmodel.__cls__()
        torchmodel = lightningmodel.model.__cls__()
        torchmodel_kwargs = lightningmodel.model.kwargs
    else:
        assert torchmodel, f'If not passing an initialized lightningmodel you must pass a torchmodel.'
        model = lightningmodel
        if not isinstance(torchmodel, type):
            assert '__cls__' in [
                f for f in dir(torchmodel.__cls__()) if callable(getattr(torchmodel.__cls__(), f))
            ],  f'Lightning Model {torchmodel.__name__} must have `__cls__` methods implemented'
            torchmodel_kwargs = torchmodel.kwargs
            torchmodel = torchmodel.__cls__()
        else:
            assert torchmodel_kwargs, f'If not passing an initialized torchmodel you must pass a torchmodel_kwargs.'

    model = model.load(ckpt_dir, model_name,
                       torchmodel=torchmodel,
                       torchmodel_kwargs=torchmodel_kwargs)
    return model


def load_pt(pt_dir: str, model_name: str,
            lightningmodel: Union[pl.LightningModule, type]) -> pl.LightningModule:
    """
    Load model from .pt file
    :param pt_dir: Checkpoint directory
    :param model_name: Model name identifier
    :param lightningmodel: Lightning Module model wrapper
    :return: Model with trained weights
    """
    loaded_model = torch.load(os.path.join(pt_dir,
                                           model_name+'.pt'))
    assert '__cls__' in [
        f for f in dir(lightningmodel.__cls__()) if callable(getattr(lightningmodel.__cls__(), f))
    ], f'Lightning Model {lightningmodel.__name__} must have `__cls__` methods implemented'
    if not isinstance(lightningmodel, type):
        lightningmodel = lightningmodel.__cls__()
    return lightningmodel(model=loaded_model.model)


def load_model(directory: str, model_name: str,
               lightningmodel: Union[pl.LightningModule, type],
               torchmodel: Optional[Union[torch.nn.Module, type]] = None,
               torchmodel_kwargs: Optional[dict] = None) -> pl.LightningModule:
    """
    Load model in any way
    :param directory: ckpt or pt file directory
    :param model_name: Model name identifier
    :param lightningmodel: Lightning Module model wrapper
    :param torchmodel: Neural Network Architecture
    :param torchmodel_kwargs: Keyword arguments for initializing Neural Network
    :return: Model with trained weights
    """
    try:
        return load_pt(pt_dir=directory, model_name=model_name, lightningmodel=lightningmodel)
    except FileNotFoundError as e:
        print(f'Could not found .pt file. {str(e)}')
        return load_ckpt(ckpt_dir=directory, model_name=model_name, lightningmodel=lightningmodel,
                         torchmodel=torchmodel, torchmodel_kwargs=torchmodel_kwargs)
