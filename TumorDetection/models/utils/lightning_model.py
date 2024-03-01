import lightning.pytorch as pl
import torch
import torch.nn.functional as tfn
from torch.optim.lr_scheduler import PolynomialLR

from TumorDetection.utils.base import BaseClass
from TumorDetection.utils.dict_classes import (LightningModelInit, EFSNetInit,
                                               OptimizerParams, PolyLRParams, Device)


class LightningModel(pl.LightningModule, BaseClass):
    """
    Base Model wrapper for using EFSNet model
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
        self.loss_seg = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor([kwargs.get('pos_weight')] * 2, device=self.device)[:, None, None]
        )
        self.loss_lab = torch.nn.CrossEntropyLoss(
            ignore_index=kwargs.get('ignore_index'),
            weight=torch.as_tensor(kwargs.get('class_weights'),
                                   device=Device.get('device')))
        self.metrics = kwargs.get('metrics')
        self.optimizer = kwargs.get('optimizer')
        self.monitor = kwargs.get('monitor')
        self.frequency = kwargs.get('frequency')
        self.resume_training = kwargs.get('resume_training')
        if isinstance(model, type):
            # Not initialized class
            nn_kwargs = self._default_config(EFSNetInit, **kwargs.get('model_kwargs'))
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

    def shared_eval_step(self, batch, name):
        """
        Shared computation across training, validation, evaluation and testing.
        :param batch:
        :param name:
        :return:
        """
        seg_logits, lab_logits = self.model.forward(batch[0])
        loss_seg = self.loss_seg(seg_logits[torch.where(batch[2] > 0)],
                                 batch[1][torch.where(batch[2] > 0)])
        loss_lab = self.loss_lab(lab_logits, batch[2])
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                '_'.join([name, metric, 'seg']): self.metrics.get(metric)(
                    torch.argmax(seg_logits, 1),
                    torch.argmax(batch[1], 1).to(torch.int),
                    task='binary')
            })
            metrics.update({
                '_'.join([name, metric, 'lab']): self.metrics.get(metric)(
                    torch.argmax(lab_logits, -1),
                    batch[2].to(torch.int),
                    task='multiclass',
                    num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"{name}_loss": loss_lab + loss_seg}}
        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training step. Overriden.
        """
        metrics = self.shared_eval_step(batch, 'train')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return metrics['train_loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step. Overriden.
        """
        metrics = self.shared_eval_step(batch, 'val')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return metrics['val_loss']

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        Test step. Overriden.
        """
        metrics = self.shared_eval_step(batch, 'test')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return metrics['test_loss']

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_pred_logits, _ = torch.argmax(self.model(batch), -1)
        y_pred = torch.argmax(y_pred_logits, -1)
        return y_pred

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **OptimizerParams.to_dict())
        scheduler = {
            "scheduler": PolynomialLR(optimizer, **PolyLRParams.to_dict()),  # TODO Find better Scheduler
            "monitor": self.monitor,
            "frequency": self.frequency
        }
        return [optimizer], [scheduler]
