import os.path
from typing import Union
import lightning.pytorch as pl
import torch.nn
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import warnings

from TumorDetection.utils.base import BaseClass
from TumorDetection.utils.dict_classes import (TrainerInit, TrainerCall, ModelCkptDir,
                                               LightningTrainerParms, LightningModelInit)
from models.utils.lightning_model import LightningModel
from models.utils.callbacks import ProgressBar


class Trainer(BaseClass):
    """
    Trainer wrapper for torch using pytorch Lightning.
    """
    def __init__(self, **kwargs):
        """
        Trainer class constructor
        :keyword lightning_trainer_params: (dict, LightningTrainerParms)
            Keyword arguments to use in ligning trainer.
        :keyword use_modelcheckpoint: (bool)
            Use model_checkpoint or not.
        :keyword model_name: (str)
            Lightning Model model_name.
        :keyword logger: (bool)
            Use TensorBoard logger ot not
        """
        kwargs = self._default_config(TrainerInit, **kwargs)
        self.trainer_params = self._default_config(LightningTrainerParms, **kwargs.get('lightning_trainer_params'))
        self.use_modelcheckpoint = kwargs.get('use_modelcheckpoint')
        callbacks = [ProgressBar(),
                     LearningRateMonitor(logging_interval='epoch')]
        if self.use_modelcheckpoint:
            callbacks.append(ModelCheckpoint(
                dirpath=ModelCkptDir.get('ckpt_dir'),
                filename=kwargs.get('model_name'),
                monitor="val_loss"
            ))
        if kwargs.get('logger'):
            os.makedirs(os.path.abspath(os.path.join(ModelCkptDir.get('ckpt_dir'), kwargs.get('model_name'))),
                        exist_ok=True)
            logger = TensorBoardLogger(save_dir=ModelCkptDir.get('ckpt_dir'), name=kwargs.get('model_name'))
        else:
            logger = False
        self.trainer = pl.Trainer(callbacks=callbacks,
                                  logger=logger,
                                  **self.trainer_params)

    def __call__(self, model: Union[pl.LightningModule, torch.nn.Module, type],
                 train_data: Union[Dataset, DataLoader],
                 test_data: Union[Dataset, DataLoader], **kwargs) -> pl.LightningModule:
        """
        Trainer main function.
        :param model: Neural Network to use.
        :param train_data: Train data to be fed
        :param test_data: Test/Validation data to evaluate.
        :keyword model_kwargs: (dict, LightningModelInit)
            Lightning Model kweyword arguments.
        :keyword verbose: (int, 1)
            Verbosity level.
        :keyword summary_depth: (int, 3)
            Depth of the torchinfo.summary()
        :keyword batch_size: (int, 32)
            Batch_size to use.
        :keyword resume_training: (bool, False)
            Whether to resume training from checkpoint or not.
        """
        kwargs = self._default_config(TrainerCall, **kwargs)
        if not isinstance(model, pl.LightningModule):
            model_kwargs = self._default_config(LightningModelInit, **kwargs.get('model_kwargs'))
            self.model = LightningModel(model, **model_kwargs)
        else:
            # Initialized class
            self.model = model
        if kwargs.get('verbose') > 0:
            summary(model, self.model.model.input_shape, batch_dim=0,
                    col_names=("input_size", "output_size", "num_params", "params_percent"),
                    depth=kwargs.get('summary_depth'),
                    row_settings=["var_names"],
                    device=self.model.device,
                    verbose=1)
        if isinstance(train_data, Dataset):
            train_data = DataLoader(train_data,
                                    batch_size=kwargs.get('batch_size'),
                                    drop_last=True,
                                    shuffle=True)
        if isinstance(test_data, Dataset):
            test_data = DataLoader(test_data,
                                   batch_size=kwargs.get('batch_size'),
                                   drop_last=True,
                                   shuffle=False)
        if kwargs.get('verbose') > 0:
            print(f'Training using device: {self.trainer.accelerator}')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    r'.*does not have many workers.*|'
                                    r'.*exists and is not empty.*|'
                                    r'.*that has Tensor Cores.*')
            self.trainer.fit(self.model,
                             train_data, test_data,
                             ckpt_path=ModelCkptDir.get(
                                 'ckpt_dir') + f'{self.model.model_name}' if kwargs.get('resume_training') else None)
        return self.model
