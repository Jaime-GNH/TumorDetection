import os
from typing import Tuple, List
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from TumorDetection.utils.base import BaseClass
from TumorDetection.utils.dict_classes import (LightningModelInit, EFSNetInit,
                                               OptimizerParams, PolyLRParams, Device)


class LightningModel(pl.LightningModule, BaseClass):
    """
    Base Model wrapper for using EFSNet model
    """
    def __init__(self, model: [torch.nn.Module, type], **kwargs):
        """
        Class constructor
        :param model: EFSNet Neural network initialized or not
        :keyword model_name: (str, 'EFSNet')
            Model name identifier.
        :keyword description: (Optional[str], 'EFSNet')
            Description of current model.
        :keyword pos_weight: (Union[int, float], 5)
            Positive weight in segmentation binary crossentropy loss.
        :keyword ignore_index: (int, -100)
            Ignore class in classifier crossentropy loss
        :keyword class_weights: (Tuple[Union[int, float],...])
            Class Weight to give to classfier crosentropy loss.
        :keyword metrics: (dict[str, Callable], {})
            Metrics to watch.
        :keyword optimizer: (torch.optim, torch.optim.Adam)
            Optimizer to use.
        :keyword monitor: (str, 'val_loss')
            Metric to monitor while training
        :keyword frequency: (int, 1)
            Number of steps until monitoring again.
        :keyword resume_training: (bool, False)
            Resume training or not.
        :keyword model_kwargs: (Optional[dict], EFSNetInit)
            Params of EFSNet model if not built.
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
        self.optimizer_params = self._default_config(OptimizerParams, **kwargs.get('optimizer_params'))
        self.scheduler_params = self._default_config(PolyLRParams, **kwargs.get('scheduler_params'))
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

    @classmethod
    def load(cls, ckpt_dir: str, model_name: str,
             model: type, nn_kwargs: dict) -> pl.LightningModule:
        pretrained_filename = os.path.join(ckpt_dir, model_name + '.ckpt')
        if os.path.isfile(pretrained_filename):
            print(f'Found pretrained model: {os.path.basename(pretrained_filename)}')
            return cls.load_from_checkpoint(pretrained_filename, model=model,
                                            model_kwargs=nn_kwargs)
        else:
            raise ValueError(f'Could not find pretrained checkpoint: {pretrained_filename}')

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs forward
        :param batch: torch images batch
        :return: tuple of predictions for batch.
        """
        return self.model(batch)

    def shared_eval_step(self, batch: List[torch.Tensor], name: str) -> dict:
        """
        Shared computation across training, validation, evaluation and testing.
        :param batch: List with evaluation batch as [images tensor, masks tensor, classes tensor]
        :param name: Name of the step.
        :return: Metrics evaluations for the step.
        """
        seg_logits, lab_logits = self.model.forward(batch[0])
        seg_loss_mask = torch.where(torch.as_tensor(batch[2] > 0))
        loss_seg = self.loss_seg(seg_logits[seg_loss_mask],
                                 batch[1][seg_loss_mask])
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

    def training_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Training step. Overriden.
        :param batch: Batch to pass.
        :param batch_idx: Use in overriden function.
        """
        metrics = self.shared_eval_step(batch, 'train')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return metrics['train_loss']

    @torch.no_grad()
    def validation_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Validation step. Overriden.
        :param batch: Batch to pass.
        :param batch_idx: Use in overriden function.
        """
        metrics = self.shared_eval_step(batch, 'val')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return metrics['val_loss']

    @torch.no_grad()
    def test_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Test step. Overriden.
        :param batch: Batch to pass.
        :param batch_idx: Use in overriden function.
        """
        metrics = self.shared_eval_step(batch, 'test')
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return metrics['test_loss']

    @torch.no_grad()
    def predict_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor,
                     dataloader_idx: int = 0) -> torch.Tensor:
        """
        Prediction step. Overriden.
        :param batch: Batch to pass.
        :param batch_idx: Use in overriden function.
        :param dataloader_idx: use in overriden function
        """
        return self.model.forward(batch)

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> List:
        """

        :param dataloader: Dataloader to predict
        :return:
        """
        y_pred = []
        for batch_idx, batch in enumerate(dataloader):
            y_pred.extend(self.predict_step(batch=batch, batch_idx=batch_idx))
        return y_pred

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_params)
        scheduler = {
            "scheduler": PolynomialLR(optimizer, **self.scheduler_params),
            "monitor": self.monitor,
            "frequency": self.frequency
        }
        return [optimizer], [scheduler]
