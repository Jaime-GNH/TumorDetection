import os
os.environ['KERAS_BACKEND'] = 'torch'

from sklearn.model_selection import train_test_split

from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDataset
from TumorDetection.utils.dict_classes import DataPathDir
from models.efsnet import EFSNet, compile_model, train_model


dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
tr_paths, val_paths = train_test_split(paths, test_size=100, random_state=0, shuffle=True)
tr_td = TorchDataset(tr_paths)
val_td = TorchDataset(val_paths)

##
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(tr_td,
#                               32,
#                               shuffle=True)
# batch = next(iter(train_dataloader))
##

# Efsmodel = EFSNetModel()
# history = Efsmodel.train_model(dataset_train=tr_td,
#                                dataset_test=val_td,
#                                resume_training=False)

model = EFSNet()
model = compile_model(model, 'adam')
model.summary(expand_nested=True)
history = train_model(model, dataset_train=tr_td, dataset_test=val_td)
