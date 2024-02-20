from TumorDetection.utils.working_dir import WorkingDir
WorkingDir.set_wd()

from trash_code.Models.GNN import Trainer
from trash_code.Models.GNN import BaseGNN
from trash_code.Data.GraphData import ImageDataLoader
from TumorDetection.utils.dict_classes import DataPathDir

# train_dataloader = GraphDataLoader(DataPathDir.get('dir_path'),
#                                    mode='train'
#                                    )
# print(next(iter(train_dataloader)))
train_dataloader = ImageDataLoader(DataPathDir.get('dir_path'),
                                   mode='train'
                                   )
test_dataloader = ImageDataLoader(DataPathDir.get('dir_path'),
                                  mode='test')

trainer = Trainer()
trainer(BaseGNN, train_dataloader, test_dataloader)

# train_dataloader = ImageDataLoader(DataPathDir.get('dir_path'),
#                                    mode='train'
#                                    )
# test_dataloader = ImageDataLoader(DataPathDir.get('dir_path'),
#                                   mode='test')
#
# trainer = Trainer()
# trainer(ImageGNN, train_dataloader, test_dataloader)
