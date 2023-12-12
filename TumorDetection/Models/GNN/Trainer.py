import os.path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.profile import count_parameters, get_model_size
from torch_geometric.nn import summary
import warnings

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import TrainerInit, TrainerCall, ModelCkptDir
from TumorDetection.Models.GNN.LightningModel import LightningModel
from TumorDetection.Models.GNN.Callbacks import ProgressBar


class Trainer(BaseClass):
    """
    """
    def __init__(self, **kwargs):
        kwargs = self._default_config(TrainerInit, **kwargs)
        self.use_earlystopping = kwargs.get('use_earlystopping')
        self.earlystopping_params = kwargs.get('es_kwargs')
        self.trainer_params = kwargs.get('lightning_trainer_params')
        self.use_modelcheckpoint = kwargs.get('use_modelcheckpoint')
        callbacks = [ProgressBar(),
                     LearningRateMonitor(logging_interval='epoch')]
        if self.use_earlystopping:
            callbacks.append(EarlyStopping(**self.earlystopping_params))
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

    def __call__(self, model, train_dataloader, test_dataloader, **kwargs):
        """
        Call
        :param model:
        :param train_dataloader:
        :param test_dataloader:
        :param kwargs:
        :return:
        """
        kwargs = self._default_config(TrainerCall, **kwargs)
        if not isinstance(model, pl.LightningModule):
            self.model = LightningModel(model, **kwargs.get('model_kwargs'))
        else:
            # Initialized class
            self.model = model

        print(self.model)
        # print(summary(self.model, next(iter(train_dataloader))))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    r'.*does not have many workers.*|'
                                    r'.*exists and is not empty.*|'
                                    r'.*that has Tensor Cores.*')
            self.trainer.fit(self.model,
                             train_dataloader, test_dataloader,
                             ckpt_path=ModelCkptDir.get(
                                 'ckpt_dir') + f'{self.model.model_name}' if kwargs.get('resume_training') else None)
