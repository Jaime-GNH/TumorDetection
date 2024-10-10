import os
from typing import Any, Tuple

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset
from TumorDetection.data.split_data import stratified_split
from TumorDetection.utils.dict_classes import DataPathDir, ReportingPathDir, ClassValues, Verbosity, Device
from TumorDetection.models.efsnet import EFSNet
from TumorDetection.models.utils.lightning_model import LightningModel
from TumorDetection.models.utils.trainer import Trainer

MODEL_NAME: str = 'EFSNet'
DESCRIPTION: str = 'EFSNet'
CLASSES: Tuple[str, ...] = tuple(ClassValues.to_dict())
VERBOSE: int = Verbosity.get('verbose')
DEVICE: Any = Device.get('device')
EPOCHS: int = 2500
IGNORE_INDEX: int = 0
BATCH_SIZE: int = 64
VAL_SIZE: int = BATCH_SIZE
TEST_SIZE: int = BATCH_SIZE
KFOLD: int = 5

# GET DATA
dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
for k in range(KFOLD):
    if not os.path.exists(os.path.join(ReportingPathDir.get('dir_path'), 'ckpt', MODEL_NAME + f'_k{k}.pt')):
        tr_paths, val_paths, test_paths = stratified_split(paths=paths, classes=tuple(ClassValues.to_dict()),
                                                           test_size=TEST_SIZE, val_size=VAL_SIZE,
                                                           random_state=k, shuffle=True)

        tr_td = TorchDataset(tr_paths,
                             crop_prob=0.5,
                             rotation_degrees=30,
                             range_contrast=(0.75, 1.25),
                             range_brightness=(0.75, 1.25),
                             vertical_flip_prob=0.,
                             horizontal_flip_prob=0.25)
        val_td = TorchDataset(val_paths,
                              crop_prob=None,
                              rotation_degrees=None,
                              range_contrast=None,
                              range_brightness=None,
                              vertical_flip_prob=None,
                              horizontal_flip_prob=None)
        test_td = TorchDataset(test_paths,
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
                validation_data=val_td,
                test_data=test_td,
                from_checkpoint=True,
                validate_model=True,
                test_model=True)
