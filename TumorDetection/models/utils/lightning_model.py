import os
from typing import Optional, Union, Tuple, List
import lightning.pytorch as pl
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR
from TumorDetection.utils.base import BaseClass
from TumorDetection.utils.dict_classes import OptimizerParams, PolyLRParams, Device


class LightningModel(pl.LightningModule, BaseClass):
    """
    Base Model wrapper for using EFSNet model
    """
    def __init__(self, model: Union[torch.nn.Module, type],
                 model_params: dict = None,
                 model_name: str = 'EFSNet', description: Optional[str] = None,
                 pos_weight: int = 1, ignore_index: int = 100, class_weights: Optional[List] = None,
                 optimizer: Union[partial, type] = partial(torch.optim.Adam, **OptimizerParams.to_dict()),
                 scheduler: Optional[Union[partial, type]] = partial(PolynomialLR, **PolyLRParams.to_dict()),
                 monitor: Optional[str] = 'val_loss', frequency: Optional[int] = 1,
                 resume_training: bool = False,
                 metrics: Optional[dict] = None, device: str = Device.get('device'),
                 ):
        """
        Class constructor
        :param model: EFSNet Neural network initialized or not
        :param model_params: Params of EFSNet model if not built.
        :param model_name: Model name identifier.
        :param description: Description of current model.
        :param pos_weight: Positive weight in segmentation binary crossentropy loss.
        :param ignore_index: Ignore class in classifier crossentropy loss
        :param class_weights: Class Weight to give to classfier crosentropy loss.
        :param optimizer: partial scheduler initialization. Anything unless the params.
        :param scheduler: partial scheduler initialization. Anything unless the optimizer param.
        :param monitor: metric to monitor with scheduler
        :param frequency: Number of steps until monitoring again.
        :param resume_training:
        :param metrics: Metrics to watch.
        :param device: device to use for computing
        """
        super().__init__()
        self.model_name = model_name
        self.description = description
        self.loss_seg = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor([pos_weight] * 2, device=self.device)[:, None, None]
        )
        self.loss_lab = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=torch.as_tensor(class_weights,
                                   device=device) if class_weights is not None else None)
        self.metrics = metrics if metrics is not None else {}
        self.optimizer = optimizer

        self.scheduler = scheduler if scheduler is not None else None
        if self.scheduler is None:
            assert all([monitor, frequency]),\
                f'If using an scheduler you must pass a valid value for monitor ({monitor}) and frequency ({frequency})'

        self.monitor = monitor
        self.frequency = frequency
        self.resume_training = resume_training
        if isinstance(model, type):
            assert model_params is not None, \
                f'If using a non-initialized model you must pass a valid value for model_params. Got: None'
            self.model = model(**model_params)
        else:
            # Initialized class
            self.model = model
        self.save_hyperparameters(ignore=['model', 'loss_seg', 'loss_lab', 'metrics',
                                          'optimizer', 'scheduler', 'monitor',
                                          'frequency', 'resume_training'])

    @classmethod
    def load(cls, ckpt_dir: str, model_name: str,
             torchmodel: type, torchmodel_kwargs: dict) -> pl.LightningModule:
        """
        Load model from checkpoint (.ckpt) file
        :param ckpt_dir: checkpoint directory
        :param model_name: Model Name Identifier
        :param torchmodel: Neural Network Architecture
        :param torchmodel_kwargs: keyword arguments
        :return:
        """
        pretrained_filename = os.path.join(ckpt_dir, model_name + '.ckpt')
        if os.path.isfile(pretrained_filename):
            print(f'Found pretrained model: {os.path.basename(pretrained_filename)}')
            return cls.load_from_checkpoint(pretrained_filename, model=torchmodel,
                                            model_params=torchmodel_kwargs)
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
        metrics = {**metrics,
                   **{f"{name}_loss_lab": loss_lab,
                      f"{name}_loss_seg": loss_seg,
                      f"{name}_loss": loss_lab + loss_seg}}
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
    def predict_dataloader(self, dataloader: DataLoader) -> List:
        """

        :param dataloader: Dataloader to predict
        :return:
        """
        y_pred = []
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, list):
                y_pred.extend(self.predict_step(batch=batch[0], batch_idx=torch.as_tensor(batch_idx)))
            else:
                y_pred.extend(self.predict_step(batch=batch, batch_idx=torch.as_tensor(batch_idx)))
        return y_pred

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())
        if self.scheduler is not None:
            scheduler = {
                "scheduler": self.scheduler(optimizer=optimizer),
                "monitor": self.monitor,
                "frequency": self.frequency
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]
