import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi, get_data_size

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import (LightningModelInit, BaseGNNInit,
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
            gnn_kwargs = self._default_config(BaseGNNInit, **kwargs.get('gnn_kwargs'))
            self.model = model(**gnn_kwargs)
        else:
            # Initialized class
            self.model = model
        # try:
        #     self.model = torch.compile(self.model)
        # except RuntimeError:
        #     pass

    def training_step(self, batch, batch_idx):
        """
        Training step. Overriden.
        """
        y_hat = self.model(batch)
        ratio = (torch.count_nonzero(batch.y == 1, dim=0)/batch.y.size(0)).to(self.device)
        loss = ratio*self.criterion(y_hat, batch.y)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'train_' + metric: self.metrics.get(metric)(y_hat, batch.y, task='multiclass',
                                                             num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"train_loss": loss}}
        self.log('Data Mb: ', float(get_data_size(batch)), on_step=True)
        try:
            self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True)
        except Exception as e:
            pass
        for k, v in metrics.items():
            self.log(k, v.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Overriden.
        """
        y_hat = self.model(batch)
        ratio = (torch.count_nonzero(batch.y == 1, dim=0) / batch.y.size(0)).to(self.device)
        loss = self.criterion(y_hat, batch.y)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'val_' + metric: self.metrics.get(metric)(y_hat, batch.y, task='multiclass',
                                                           num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"val_loss": loss}}
        self.log('Data Mb: ', float(get_data_size(batch)), on_step=True)
        try:
            self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True)
        except Exception as e:
            pass
        for k, v in metrics.items():
            self.log(k, v.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step. Overriden.
        """
        y_hat = self.model(batch)
        loss = self.criterion(y_hat, batch.y)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'test_' + metric: self.metrics.get(metric)(y_hat, batch.y, task='multiclass',
                                                            num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"test_loss": loss}}
        self.log('Data Mb: ', float(get_data_size(batch)), on_step=True)
        try:
            self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True)
        except Exception as e:
            pass
        for k, v in metrics.items():
            self.log(k, v.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch.size(0))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self.model(batch)

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
