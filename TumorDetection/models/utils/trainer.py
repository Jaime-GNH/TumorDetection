import os.path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import warnings

from TumorDetection.utils.base import BaseClass
from TumorDetection.utils.dict_classes import TrainerInit, TrainerCall, ModelCkptDir
from models.utils.lightning_model import LightningModel
from models.utils.callbacks import ProgressBar


class Trainer(BaseClass):
    """
    """
    def __init__(self, **kwargs):
        kwargs = self._default_config(TrainerInit, **kwargs)
        self.trainer_params = kwargs.get('lightning_trainer_params')
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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    r'.*does not have many workers.*|'
                                    r'.*exists and is not empty.*|'
                                    r'.*that has Tensor Cores.*')
            self.trainer.fit(self.model,
                             train_dataloader, test_dataloader,
                             ckpt_path=ModelCkptDir.get(
                                 'ckpt_dir') + f'{self.model.model_name}' if kwargs.get('resume_training') else None)
