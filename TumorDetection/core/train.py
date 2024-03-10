import os
from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset
from TumorDetection.utils.dict_classes import DataPathDir, ReportingPathDir, Verbosity, Device
from TumorDetection.models.efsnet import EFSNet
from TumorDetection.models.utils.lightning_model import LightningModel
from TumorDetection.models.utils.trainer import Trainer

MODEL_NAME = 'EFSNet_clf_seg'
DESCRIPTION = 'EFSNet with classification and binary segmentation.'
CLASS_WEIGHTS = [1., 3., 3.]
POS_WEIGHT = 5
VERBOSE = Verbosity.get('verbose')
DEVICE = Device.get('device')

dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=100, random_state=0, shuffle=True)
tr_td = TorchDataset(tr_paths)
val_td = TorchDataset(val_paths)

trainer = Trainer(model_name=MODEL_NAME,
                  ckpt_dir=os.path.join(ReportingPathDir.get('dir_path'), 'ckpt'),
                  verbose=VERBOSE)
lighningmodel = LightningModel(model=EFSNet(device=DEVICE,
                                            verbose=VERBOSE),
                               model_name=MODEL_NAME,
                               description=DESCRIPTION,
                               class_weights=CLASS_WEIGHTS,
                               pos_weight=POS_WEIGHT,
                               device=DEVICE)
trainer(model=lighningmodel,
        train_data=tr_td,
        test_data=val_td,
        from_checkpoint=False,
        validate_model=True,
        test_model=True)
