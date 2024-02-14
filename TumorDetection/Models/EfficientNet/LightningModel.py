import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import (LightningModelInit, EfficientNetInit,
                                              OptimizerParams)


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
        self.description = kwargs.get('description')
        self.criterion = kwargs.get('criterion')
        self.metrics = kwargs.get('metrics')
        self.optimizer = kwargs.get('optimizer')
        self.resume_training = kwargs.get('resume_training')
        self.reducelronplateau_params = kwargs.get('rlr_kwargs')
        self.use_reducelronplateau = kwargs.get('use_reducelronplateau')
        if kwargs.get('save_hyperparameters'):
            self.save_hyperparameters(ignore=['criterion'])
        if isinstance(model, type):
            # Not initialized class
            nn_kwargs = self._default_config(EfficientNetInit, **kwargs.get('nn_kwargs'))
            self.model = model(**nn_kwargs)
        else:
            # Initialized class
            self.model = model

    def forward(self, batch):
        """
        Runs forward
        :param batch:
        :return:
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step. Overriden.
        """
        y_pred_logits = self.model.forward(batch)
        y_true = batch['mask']
        loss = self.criterion(y_pred_logits, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'train_' + metric: self.metrics.get(metric)(torch.argmax(y_pred_logits, -1),
                                                             y_true.to(torch.int),
                                                             task='multiclass',
                                                             num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"train_loss": loss}}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step. Overriden.
        """
        y_pred_logits = self.model(batch)
        y_true = batch['mask']
        loss = self.criterion(y_pred_logits, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'val_' + metric: self.metrics.get(metric)(torch.argmax(y_pred_logits, -1),
                                                           y_true.to(torch.int),
                                                           task='multiclass',
                                                           num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"val_loss": loss}}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        Test step. Overriden.
        """
        y_pred_logits = self.model(batch)
        y_true = batch['mask']
        loss = self.criterion(y_pred_logits, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'test_' + metric: self.metrics.get(metric)(torch.argmax(y_pred_logits, -1),
                                                            y_true.to(torch.int),
                                                            task='multiclass',
                                                            num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"test_loss": loss}}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_pred_logits, _ = torch.argmax(self.model(batch), -1)
        y_pred = torch.argmax(y_pred_logits, -1)
        return y_pred

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **OptimizerParams.to_dict())
        if self.use_reducelronplateau:
            scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, **{k: v for k, v in self.reducelronplateau_params.items()
                                                             if k not in ['frequency', 'monitor']}),
                "monitor": self.reducelronplateau_params['monitor'],
                "frequency": self.reducelronplateau_params['frequency']
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]
