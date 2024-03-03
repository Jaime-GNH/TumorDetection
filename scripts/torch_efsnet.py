from time import perf_counter
import torch
from torchinfo import summary
from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset, Dataset
from TumorDetection.utils.dict_classes import DataPathDir, Device
from TumorDetection.models.efsnet import EFSNet


dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=100, random_state=0, shuffle=True)
tr_td = TorchDataset(tr_paths)
val_td = TorchDataset(val_paths)
print(isinstance(tr_td, Dataset))
print(isinstance(tr_td, TorchDataset))
#
from torch.utils.data import DataLoader
train_dataloader = DataLoader(tr_td,
                              32,
                              shuffle=True)
batch = next(iter(train_dataloader))

model = EFSNet()
summary(model, (1, 256, 256), batch_dim=0,
        col_names=("input_size", "output_size", "num_params", "params_percent"),
        depth=3,
        row_settings=["var_names"],
        device=Device.get('device'),
        verbose=1)
print(f'Device: {model.device}')
seg, lab = model(torch.rand((1, 1, 256, 256)).to(device=Device.get('device')))
images = 1000
it = perf_counter()
for _ in range(images):
    model(torch.rand((1, 1, 256, 256)).to(device=Device.get('device')))
et = perf_counter()
print(f'Time per 100 images: {et-it:.3f}')
print(f'FPS: {images/(et-it):.3f}')
