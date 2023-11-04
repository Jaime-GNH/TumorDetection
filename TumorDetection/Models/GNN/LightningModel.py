import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.types import ReduceLROnPlateau

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import (LightningModelInit, HyperGNNInit, TrainerInit, ModelCkptDir,
                                              Verbosity, OptimizerParams)


class LightningModel(pl.LightningModule, BaseClass):
    """
    Base Model wrapper for using any torch model
    """
    def __init__(self, model, **kwargs):
        """
        Class constructor
        :param model: (torch.Module)
            Neural network initialized or not
        :param kwargs: (dict)
            GNNModelInit attributes.
        """
        super().__init__()
        kwargs = self._default_config(LightningModelInit, **kwargs)
        self.model_name = kwargs.get('model_name')
        self.criterion = kwargs.get('criterion')
        self.metrics = kwargs.get('metrics')
        self.use_earlystopping = kwargs.get('use_earlystopping')
        self.earlystopping_params = kwargs.get('es_kwargs')
        self.reducelronplateau_params = kwargs.get('rlr_kwargs')
        self.use_modelcheckpoint = kwargs.get('use_modelcheckpoint')
        self.use_reducelronplateau = kwargs.get('use_reducelronplateau')
        self.optimizer = kwargs.get('optimizer')
        self.resume_training = kwargs.get('resume_training')
        if kwargs.get('save_hyperparameters'):
            self.save_hyperparameters()
        if isinstance(model, type):
            # Not initialized class
            model_kwargs = self._default_config(HyperGNNInit, **kwargs.get('model_kwargs'))
            self.model = model(**model_kwargs)
        else:
            # Initialized class
            self.model = model

    def forward(self, x):
        """
        Forward inference pass
        :param x:
        :return:
        """
        # TODO: This is the forward pass for inference. Should lead into a mask image.
        #  By now it leads to a graph representation.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step. Overriden.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Overriden.
        """
        self._shared_eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        """
        Test step. Overriden.
        """
        self._shared_eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self(batch)

    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'{prefix}_'+metric: self.metrics.get(metric)(y_hat, y)
            })
        metrics = {**metrics, **{f"{prefix}_loss": loss}}
        self.log_dict(metrics)

    def configure_callbacks(self):
        callbacks = []
        if self.use_earlystopping:
            callbacks.append(EarlyStopping(**self.earlystopping_params))
        if self.use_modelcheckpoint:
            callbacks.append(ModelCheckpoint(
                dirpath=ModelCkptDir.get('ckpt_dir'),
                filename=f'{self.model_name}',
                monitor="val_loss"
            ))
        return callbacks

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **OptimizerParams.to_dict())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, **self.reducelronplateau_params),
                "monitor": self.reducelronplateau_params['monitor'],
            } if self.use_reducelronplateau else {}
        }

    def set_trainer(self, **kwargs):
        kwargs = self._default_config(TrainerInit, **kwargs)
        self.trainer = pl.Trainer(**kwargs)

    def fit(self, train_dataloaders=None, val_dataloaders=None, **kwargs):
        """
        Fit the model
        :param train_dataloaders: (torch.DataLoader)
        :param val_dataloaders: (torch.DataLoader)
        :param kwargs: (dict)
        :return:
        """
        if not hasattr(self, 'trainer'):
            self.set_trainer(**kwargs)
        self.trainer.fit(self.model, train_dataloaders, val_dataloaders,
                         ckpt_path=ModelCkptDir.get('ckpt_dir')+f'{self.model_name}' if self.resume_training else None)

    def validate(self, dataloaders, **kwargs):
        """
        Validate model
        :param dataloaders:
        :param kwargs:
        :return:
        """
        if not hasattr(self, 'trainer'):
            self.set_trainer(**kwargs)
        self.trainer.validate(self.model, dataloaders,
                              ckpt_path=ModelCkptDir.get('ckpt_dir')+f'{self.model_name}',
                              verbose=Verbosity.get('verbose') > 0)

    def test(self, dataloaders, **kwargs):
        """
        Validate model
        :param dataloaders:
        :param kwargs:
        :return:
        """
        if not hasattr(self, 'trainer'):
            self.set_trainer(**kwargs)
        self.trainer.test(self.model, dataloaders,
                          ckpt_path=ModelCkptDir.get('ckpt_dir') + f'{self.model_name}',
                          verbose=Verbosity.get('verbose') > 0)
