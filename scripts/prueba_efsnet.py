import os
os.environ['KERAS_BACKEND'] = 'torch'

from time import perf_counter
import torch

from TumorDetection.Utils.DictClasses import DataPathDir
from TumorDetection.Data.Loader import DataPathLoader, ImageLoader
from TumorDetection.Data.Dataset import PatchDataset
from TumorDetection.Models.EfficientNet.EfficientNet import EFSNetModel

dp = DataPathLoader(DataPathDir.get('dir_path'),
                    substring_filter='normal')
il = ImageLoader()(paths_classes=dp())

Efsmodel = EFSNetModel(verbose=0, input_shape=(512, 1024, 3))
Efsmodel.compile_model()
print('Params:', Efsmodel.model.count_params())
print('Torch params:', sum(p.numel() for p in Efsmodel.model.parameters()))
print('Torch trainable params:', sum(p.numel() for p in Efsmodel.model.parameters() if p.requires_grad))
print('Device:', next(iter(Efsmodel.model.parameters())).device)
# TIEMPO

x = torch.rand((1, 512, 1024, 3))
it = perf_counter()
for i in range(100):
    y = Efsmodel.model(x)
print(f'Time per image: {round((perf_counter()-it)/100, 7)} s')

# DATOS
X_tr, X_val, y_tr, y_val = PatchDataset()(images=[t[2] for t in il],
                                          masks=[t[-1] for t in il])
print('Train data')
print(X_tr.shape, y_tr.shape)
print('Validation data')
print(X_val.shape, y_val.shape)

# TRAIN

Efsmodel = EFSNetModel()
history = Efsmodel.train_model(X_tr, y_tr, X_val, y_val, resume_training=False)
# history = Efsmodel.train_model(torch.from_numpy(X_tr),
#                                torch.from_numpy(y_tr),
#                                torch.from_numpy(X_val),
#                                torch.from_numpy(y_val),
#                                resume_training=False)

