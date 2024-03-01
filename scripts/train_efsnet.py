from time import perf_counter
import torch

from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset
from TumorDetection.utils.dict_classes import DataPathDir, Device
from TumorDetection.models.efsnet import EFSNet
from TumorDetection.models.utils.trainer import Trainer


dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=100, random_state=0, shuffle=True)
tr_td = TorchDataset(tr_paths)
val_td = TorchDataset(val_paths)

trainer = Trainer()
trainer(model=EFSNet(),
        train_data=tr_td,
        test_data=val_td)

