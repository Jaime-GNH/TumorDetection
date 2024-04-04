import os
from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDatasetClfSeg
from TumorDetection.utils.dict_classes import DataPathDir, ReportingPathDir, Verbosity, Device
from TumorDetection.models.efsnet import EFSNetClfSeg
from TumorDetection.models.utils.lightning_model import LightningModelClfSeg
from TumorDetection.models.utils.trainer import Trainer

MODEL_NAME = 'EFSNet_clf_seg'
CLASS_WEIGHTS = [1., 3., 3.]
POS_WEIGHT = 5

DESCRIPTION = ('EFSNet with classification and binary segmentation.\n'
               f'Class_weights: {CLASS_WEIGHTS}\n'
               f'Pos_weight: {POS_WEIGHT}')
VERBOSE = Verbosity.get('verbose')
DEVICE = Device.get('device')
EPOCHS = 10000
BATCH_SIZE = 64
TEST_SIZE = 100

# GET DATA
dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=TEST_SIZE, random_state=0, shuffle=True)
tr_td = TorchDatasetClfSeg(tr_paths,
                           crop_prob=0.5,
                           shear=25,
                           rotation_degrees=180,
                           range_contrast=None,
                           range_brightness=None,
                           range_saturation=None,
                           vertical_flip_prob=0.25,
                           horizontal_flip_prob=0.25)
val_td = TorchDatasetClfSeg(val_paths,
                            crop_prob=None,
                            shear=None,
                            rotation_degrees=None,
                            range_contrast=None,
                            range_brightness=None,
                            range_saturation=None,
                            vertical_flip_prob=None,
                            horizontal_flip_prob=None)

# BUILD MODEL
lighningmodel = LightningModelClfSeg(model=EFSNetClfSeg(device=DEVICE,
                                                        verbose=VERBOSE),
                                     model_name=MODEL_NAME,
                                     description=DESCRIPTION,
                                     class_weights=CLASS_WEIGHTS,
                                     pos_weight=POS_WEIGHT,
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
        from_checkpoint=False,
        validate_model=True,
        test_model=True)
