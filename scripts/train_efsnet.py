import os
from typing import Any
from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset
from TumorDetection.utils.dict_classes import DataPathDir, ReportingPathDir, Verbosity, Device
from TumorDetection.models.efsnet import EFSNet
from TumorDetection.models.utils.lightning_model import LightningModel
from TumorDetection.models.utils.trainer import Trainer

MODEL_NAME: str = 'EFSNet_base'
DESCRIPTION: str = 'EFSNet base'
VERBOSE: int = Verbosity.get('verbose')
DEVICE: Any = Device.get('device')
EPOCHS: int = 2500
# CLASS_WEIGHT = [1., 5., 5.]
IGNORE_INDEX: int = 0
BATCH_SIZE: int = 64
TEST_SIZE: int = 100

# GET DATA
dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=TEST_SIZE, random_state=0, shuffle=True)
tr_td = TorchDataset(tr_paths,
                     crop_prob=0.5,
                     rotation_degrees=180,
                     range_contrast=(0.75, 1.25),
                     range_brightness=(0.75, 1.25),
                     vertical_flip_prob=0.25,
                     horizontal_flip_prob=0.25)
val_td = TorchDataset(val_paths,
                      crop_prob=None,
                      rotation_degrees=None,
                      range_contrast=None,
                      range_brightness=None,
                      vertical_flip_prob=None,
                      horizontal_flip_prob=None)

# BUILD MODEL
lighningmodel = LightningModel(model=EFSNet(device=DEVICE,
                                            verbose=VERBOSE),
                               model_name=MODEL_NAME,
                               description=DESCRIPTION,
                               ignore_index=IGNORE_INDEX,
                               # class_weights=CLASS_WEIGHT,
                               device=DEVICE)

# TRAIN
trainer = Trainer(model_name=MODEL_NAME,
                  max_epochs=EPOCHS,
                  ckpt_dir=os.path.join(ReportingPathDir.get('dir_path'), 'ckpt'),
                  verbose=VERBOSE)
trainer(model=lighningmodel,
        train_batch_size=BATCH_SIZE,
        val_batch_size=TEST_SIZE,
        train_data=tr_td,
        test_data=val_td,
        from_checkpoint=True,
        validate_model=True,
        test_model=True)
