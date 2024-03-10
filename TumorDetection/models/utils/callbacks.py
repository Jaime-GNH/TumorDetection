import sys
import os
from typing import Dict, Any, Optional
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class ProgressBar(TQDMProgressBar):
    """
    Progress Bar Callback for PyCharm terminal.
    """
    def init_validation_tqdm(self) -> Tqdm:
        """
        Initializes Validation bar and disables it if program is not running on a terminal.
        :return: Progress Bar
        """
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """
        Initializes Predict bar and disables it if program is not running on a terminal.
        :return: Progress Bar
        """
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """
        Initializes Test bar and disables it if program is not running on a terminal.
        :return: Progress Bar
        """
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class PtModelCheckpoint(ModelCheckpoint):
    """
    Checkpoint model on pt
    """
    def __init__(self, dirpath: Optional[str], filename: Optional[str],
                 monitor: Optional[str], mode: Optional[str],
                 save_weights_only: Optional[bool] = False, verbose: Optional[bool] = False,
                 enable_version_counter: bool = False,
                 suffix: Optional[str] = None):
        super().__init__(dirpath=dirpath, filename=filename, monitor=monitor,
                         mode=mode, save_weights_only=save_weights_only,
                         enable_version_counter=enable_version_counter,
                         verbose=verbose)
        self.suffix = '_' + suffix if suffix is not None else ''
        self.path = os.path.join(self.dirpath, self.filename + self.suffix + '.pt')

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        torch.save(pl_module.model,
                   self.path)
