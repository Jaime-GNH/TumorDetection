from time import perf_counter
import torch
from torchinfo import summary

from TumorDetection.utils.dict_classes import Device
from TumorDetection.models.efsnet import EFSNet

# Segmenter
model = EFSNet(input_shape=(3, 1024, 512))
summary(model, (3, 1024, 512), batch_dim=0,
        col_names=("input_size", "output_size", "num_params", "params_percent"),
        depth=2,
        row_settings=["var_names"],
        device=Device.get('device'),
        verbose=1)
print(f'Device: {model.device}')
seg = model(torch.rand((1, 1, 256, 256)).to(device=Device.get('device')))
images = 1000
it = perf_counter()
for _ in range(images):
    model(torch.rand((1, 1, 256, 256)).to(device=Device.get('device')))
et = perf_counter()
print(f'Time per 100 images: {et-it:.3f}')
print(f'FPS: {images/(et-it):.3f}')
