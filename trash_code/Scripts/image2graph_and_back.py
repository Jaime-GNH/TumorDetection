import torch_geometric as tg
from TumorDetection.utils.working_dir import WorkingDir
WorkingDir.set_wd()

from trash_code.Data.GraphData import ImageDataLoader
from TumorDetection.utils.dict_classes import DataPathDir
from trash_code.Utils.utils import apply_function2list, tup2hypergraph

train_dataloader = ImageDataLoader(DataPathDir.get('dir_path'),
                                   mode='train'
                                   )
batch = next(iter(train_dataloader))
print(batch)
new_batch = list((i, m, None) for i, m in zip(batch.get('image'), batch.get('mask')))
DataBatch = tg.data.Batch.from_data_list(
    apply_function2list(
        new_batch,
        tup2hypergraph,
        image_idx=0,
        mask_idx=1,
        current_solution_idx=2,
        hypernode_patch_dim=64,
        kernel_kind='corner'
    )
)
print(DataBatch)

batch_imgs = DataBatch.x.resize(len(DataBatch.original_shape), *DataBatch.original_shape[0])
batch_mask = (DataBatch.y.resize(len(DataBatch.original_shape),
                                 DataBatch.hypernode_patch_dim[0].item(),
                                 1)
              .repeat(1, 1, DataBatch.x.size(1))
              .resize(len(DataBatch.original_shape), *DataBatch.original_shape[0]))

# TODO: DataBatch to image, mask.
# TODO: prediction to current_solution
