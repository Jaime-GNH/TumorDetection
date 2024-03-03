import sys
from lightning.pytorch.callbacks import TQDMProgressBar
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
