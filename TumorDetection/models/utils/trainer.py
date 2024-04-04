import os.path
from typing import Union, Optional
import lightning.pytorch as pl
import torch.nn
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import warnings

from TumorDetection.utils.base import BaseClass
from TumorDetection.models.utils.callbacks import ProgressBar, PtModelCheckpoint
from TumorDetection.models.utils.load import load_model, load_ckpt


class Trainer(BaseClass):
    """
    Trainer wrapper for torch using pytorch Lightning.
    """
    def __init__(self, model_name: Optional[str] = 'EFSNet', max_epochs: int = 10001,
                 use_model_ckpt: bool = True, ckpt_dir: str = os.getcwd(),
                 force_ckpt_dir: bool = True,
                 limit_train_batches: Optional[int] = None, limit_val_batches: Optional[int] = None,
                 limit_test_batches: Optional[int] = None, gradient_clip_val: Optional[float] = None,
                 accelerator: str = 'auto', logger: bool = True, seed: Optional[int] = None,
                 verbose: int = 1):
        """
        Trainer class constructor
        :param model_name:
        :param max_epochs:
        :param use_model_ckpt:
        :param ckpt_dir:
        :param force_ckpt_dir:
        :param limit_train_batches:
        :param limit_val_batches:
        :param limit_test_batches:
        :param gradient_clip_val:
        :param accelerator:
        :param logger:
        :param seed:
        :param verbose:
        """
        self.model_name = model_name
        self.verbose = verbose
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        if force_ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        if seed is not None:
            pl.seed_everything(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if logger:
            os.makedirs(os.path.abspath(os.path.join(self.ckpt_dir, self.model_name)),
                        exist_ok=True)
            logger = TensorBoardLogger(save_dir=ckpt_dir, name=model_name)
        else:
            logger = False
        callbacks = [ProgressBar(),
                     LearningRateMonitor(logging_interval='epoch')]
        if use_model_ckpt:
            callbacks.append(ModelCheckpoint(
                dirpath=self.ckpt_dir,
                filename=self.model_name,
                monitor="val_loss", mode='min',
                enable_version_counter=False,
                verbose=self.verbose > 1
            ))
            callbacks.append(PtModelCheckpoint(
                dirpath=self.ckpt_dir,
                filename=self.model_name,
                monitor="val_loss", mode='min',
                verbose=self.verbose > 1
            ))
            self.ckpt_path = os.path.join(self.ckpt_dir, self.model_name + '.ckpt')
        self.trainer = pl.Trainer(logger=logger, callbacks=callbacks,
                                  accelerator=accelerator, max_epochs=max_epochs,
                                  enable_model_summary=False, enable_progress_bar=True,
                                  log_every_n_steps=1, num_sanity_val_steps=0,
                                  limit_train_batches=limit_train_batches,
                                  limit_val_batches=limit_val_batches,
                                  limit_test_batches=limit_test_batches,
                                  gradient_clip_val=gradient_clip_val)

    def __call__(self, model: Union[pl.LightningModule, torch.nn.Module, type],
                 train_data: Union[Dataset, DataLoader],
                 test_data: Union[Dataset, DataLoader],
                 lightningmodel_cls: Optional[type] = None,
                 lightningmodel_params: Optional[dict] = None,
                 torchmodel_params: Optional[dict] = None,
                 train_batch_size: Optional[int] = 32, val_batch_size: Optional[int] = 32,
                 from_checkpoint: bool = False,
                 summary_depth: int = 3,
                 validate_model: bool = False, test_model: bool = False) -> pl.LightningModule:
        """
        Trainer main function.
        :param model: Neural Network to use.
        :param train_data: Train data to be fed
        :param test_data: Test/Validation data to evaluate.
        :param lightningmodel_cls: LightningModule Class if using a torch.nn.Module or type
        :param lightningmodel_params: Parameters for LightningModule for initializing if needed
        :param torchmodel_params: Parameters for Torch Neural Network for initializing if needed
        :param train_batch_size: batch size for training if passing a train_data of type Dataset
        :param val_batch_size: batch size for validation and testing if passing a test_data of type Dataset
        :param from_checkpoint: load model from checkpoint
        :param summary_depth: Model summary depth to observe
        :param validate_model: Validate model previous to training
        :param test_model: Test model after training
        :return: model trained.
        """
        if isinstance(model, pl.LightningModule):
            if self.model_name != model.model_name:
                warnings.warn(f'Trainer model_name: {self.model_name} and '
                              f'LightningModule model_name: {model.model_name} are not equal.'
                              f' Cosidered as intended. Taking trainer model_name.',
                              UserWarning)
                model.model_name = self.model_name
        elif isinstance(model, (torch.nn.Module, type)):
            assert lightningmodel_cls, ('If passing a model of istance torch.nn.Module (initialized or not)'
                                        ' you must pass a lightningmodel_cls')
            if isinstance(model, type):
                warnings.warn(f'Got a type class but no torchmodule_params'
                              f'Constructing with default values. Considered as intended.',
                              UserWarning)
                model = model(**torchmodel_params)
            if lightningmodel_params is None:
                warnings.warn(f'You passed a torch.nn.Module but no lightningmodule_params.'
                              f'Constructing with default values. Considered as intended.',
                              UserWarning)

            model = lightningmodel_cls(model, lightningmodel_params)

        if self.verbose > 2:
            if torch.cuda.is_available() and self.trainer.accelerator != 'cpu':
                print('Torch is using cuda')
            else:
                print('Torch is using cpu. Low performance is expected')
                print('Torch cuda is available:', torch.cuda.is_available())
                print('Trainer Accelerator:', self.trainer.accelerator)
        if isinstance(train_data, Dataset):
            train_data = DataLoader(train_data,
                                    batch_size=train_batch_size,
                                    drop_last=True,
                                    shuffle=True)
        if isinstance(test_data, Dataset):
            test_data = DataLoader(test_data,
                                   batch_size=val_batch_size,
                                   drop_last=True,
                                   shuffle=False)
        if from_checkpoint:
            model = load_model(self.ckpt_dir, self.model_name, model)

        if self.verbose > 1:
            summary(model, model.model.input_shape, batch_dim=0,
                    col_names=("input_size", "output_size", "num_params", "params_percent"),
                    depth=summary_depth,
                    row_settings=["var_names"],
                    device=model.device,
                    verbose=min(1, self.verbose))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    r'.*does not have many workers.*|'
                                    r'.*exists and is not empty.*|'
                                    r'.*that has Tensor Cores.*')
            if validate_model:
                if self.verbose > 0:
                    print('Validating model...')
                self.trainer.validate(model, test_data, verbose=self.verbose > 0)
            if self.verbose > 0:
                print('Training model...')
            self.trainer.fit(model,
                             train_dataloaders=train_data,
                             val_dataloaders=test_data)
        torch.save(model,
                   os.path.join(self.ckpt_dir, self.model_name + '.pt'))
        model = load_ckpt(self.ckpt_dir,
                          self.model_name,
                          model)
        if test_model:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*does not have many workers.*')
                self.trainer.test(model,
                                  dataloaders=test_data,
                                  verbose=self.verbose > 0)

        return model
