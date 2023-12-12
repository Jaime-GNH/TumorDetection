import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import (LightningModelInit, ImageGNNInit,
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
            gnn_kwargs = self._default_config(ImageGNNInit, **kwargs.get('gnn_kwargs'))
            self.model = model(**gnn_kwargs)
        else:
            # Initialized class
            self.model = model
        # try:
        #     self.model = torch.compile(self.model)
        # except RuntimeError:
        #     pass

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
        losses = []
        y_preds = []
        y_trues = []
        solutions, _, node_masks = self.model(batch)
        for i, (y_hat, node_mask) in enumerate(zip(solutions, node_masks)):
            y_pred = -4 * torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
            y_pred[node_mask > 0] = y_hat.squeeze()
            y_mask = (batch.get('mask')
                      .unfold(1, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .unfold(2, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .flatten(1, 2).flatten(2).flatten(0, 1).max(1).values)
            y_true = y_mask[(y_mask > 0) | (node_mask > 0)].to(y_pred.dtype)
            y_pred = y_pred[(y_mask > 0) | (node_mask > 0)]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            losses.append(self.criterion[i](y_pred, y_true))
        numels = [torch.numel(loss)+1 for loss in losses]
        loss = sum([numel*loss for numel, loss in zip(numels, losses)])/sum(numels)
        # y_hat, _, node_mask = self.model(batch)
        # y_pred = -4*torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
        # y_pred[node_mask > 0] = y_hat.squeeze()
        # y_true = batch.get('mask').flatten()[(batch.get('mask').flatten() > 0) | (node_mask > 0)].to(y_pred.dtype)
        # y_pred = y_pred[(batch.get('mask').flatten() > 0) | (node_mask > 0)]
        # loss = self.criterion(y_pred, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'train_' + metric: self.metrics.get(metric)(torch.sigmoid(y_preds[-1]), y_trues[-1].to(torch.int),
                                                             task='binary')  #, task='multiclass',
                                                             # num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"train_loss": loss}}
        # self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True,
        #          batch_size=len(batch['image']))
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step. Overriden.
        """
        losses = []
        y_preds = []
        y_trues = []
        solutions, _, node_masks = self.model(batch)
        for i, (y_hat, node_mask) in enumerate(zip(solutions, node_masks)):
            y_pred = -4 * torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
            y_pred[node_mask > 0] = y_hat.squeeze()
            y_mask = (batch.get('mask')
                      .unfold(1, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .unfold(2, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .flatten(1, 2).flatten(2).flatten(0, 1).max(1).values)
            y_true = y_mask[(y_mask > 0) | (node_mask > 0)].to(y_pred.dtype)
            y_pred = y_pred[(y_mask > 0) | (node_mask > 0)]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            losses.append(self.criterion[i](y_pred, y_true))
        numels = [torch.numel(loss) + 1 for loss in losses]
        loss = sum([numel * loss for numel, loss in zip(numels, losses)]) / sum(numels)
        # y_hat, _, node_mask = self.model(batch)
        # y_pred = -4*torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
        # y_pred[node_mask > 0] = y_hat.squeeze()
        # y_true = batch.get('mask').flatten()[(batch.get('mask').flatten() > 0) | (node_mask > 0)].to(y_pred.dtype)
        # y_pred = y_pred[(batch.get('mask').flatten() > 0) | (node_mask > 0)]
        # loss = self.criterion(y_pred, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'val_' + metric: self.metrics.get(metric)(torch.sigmoid(y_preds[-1]), y_trues[-1].to(torch.int),
                                                           task='binary')  # , task='multiclass',
                # num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"val_loss": loss}}
        # self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True,
        #          batch_size=len(batch['image']))
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        Test step. Overriden.
        """
        losses = []
        y_preds = []
        y_trues = []
        solutions, _, node_masks = self.model(batch)
        for i, (y_hat, node_mask) in enumerate(zip(solutions, node_masks)):
            y_pred = -4 * torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
            y_pred[node_mask > 0] = y_hat.squeeze()
            y_mask = (batch.get('mask')
                      .unfold(1, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .unfold(2, self.model.hypernode_patch_dims[i], self.model.hypernode_patch_dims[i])
                      .flatten(1, 2).flatten(2).flatten(0, 1).max(1).values)
            y_true = y_mask[(y_mask > 0) | (node_mask > 0)].to(y_pred.dtype)
            y_pred = y_pred[(y_mask > 0) | (node_mask > 0)]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            losses.append(self.criterion[i](y_pred, y_true))
        numels = [torch.numel(loss) + 1 for loss in losses]
        loss = sum([numel * loss for numel, loss in zip(numels, losses)]) / sum(numels)
        # y_hat, _, node_mask = self.model(batch)
        # y_pred = -4*torch.ones(node_mask.size(), dtype=y_hat.dtype, device=y_hat.device)
        # y_pred[node_mask > 0] = y_hat.squeeze()
        # y_true = batch.get('mask').flatten()[(batch.get('mask').flatten() > 0) | (node_mask > 0)].to(y_pred.dtype)
        # y_pred = y_pred[(batch.get('mask').flatten() > 0) | (node_mask > 0)]
        # loss = self.criterion(y_pred, y_true)
        metrics = {}
        for metric in self.metrics:
            metrics.update({
                f'test_' + metric: self.metrics.get(metric)(torch.sigmoid(y_preds[-1]), y_trues[-1].to(torch.int),
                                                            task='binary')  # , task='multiclass',
                # num_classes=self.model.num_classes)
            })
        metrics = {**metrics, **{f"test_loss": loss}}
        # self.log('Nvidia usage Mb: ', get_gpu_memory_from_nvidia_smi()[1], on_step=True,
        #          batch_size=len(batch['image']))
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        solutions, graphbatch, node_masks = self.model(batch)
        return self.model.graph2image(graphbatch, batch.get('mask'), node_masks[-1],
                                      torch.sigmoid(solutions[-1]))['image']

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
