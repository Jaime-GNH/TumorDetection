from typing import Any
import os
import cv2
import torch
# from torchmetrics.functional import accuracy, jaccard_index

from TumorDetection.utils.working_dir import WorkingDir


class DictClass:
    """
    Dictionary as class
    """
    @classmethod
    def to_dict(cls) -> dict:
        """
        Class to dictionary
        :return: cls as dict
        """
        return {k: v for k, v in vars(cls).items() if not k.startswith('__')}

    @classmethod
    def get(cls, name: str) -> Any:
        """
        Getter param
        :return: Value from class.
        """
        return vars(cls)[name]


# [MAPS]
class ClassValues(DictClass):
    """
    Classes in BUSI dataset.
    """
    normal = 0
    benign = 1
    malignant = 2


class MappedClassValues(DictClass):
    """
    Mapping to get binary classification.
    """
    normal = 0
    tumor = 1


class BaseClassMap(DictClass):
    """
    Mapping from a class to another.
    """
    normal = 'normal'
    benign = 'tumor'
    malignant = 'tumor'


# [PARAMS]
class Verbosity(DictClass):
    """
    Verbosity level
    """
    verbose = 1


class Mask(DictClass):
    """
    Use Mask
    """
    mask = True


class Device(DictClass):
    """
    Device for computing torch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataPathDir(DictClass):
    """
    Data Path Directory
    """
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'Dataset_BUSI_with_GT')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'Dataset_BUSI_with_GT' in dir_name
    ][0]
    os.chdir(cw)


class ResourcesPathDir(DictClass):
    """
    Resources Path Directory
    """
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'resources')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'resources' in dir_name
    ][0]
    os.chdir(cw)


class ReportingPathDir(DictClass):
    """
    Reporting Path Directory
    """
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'reporting')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


class ModelCkptDir(DictClass):
    """
    Model Checkpoint Path Directory
    """
    cw = WorkingDir.set_wd()
    ckpt_dir = [
        os.path.join(dir_path, 'reporting/ckpt')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


class ViewerClsParams(DictClass):
    """
    cv2 module Viewer params.
    """
    win_title = 'Viewer'
    mask_colormap = cv2.COLORMAP_RAINBOW
    mask_alpha_weight = 0.3
    mask_beta_weight = 0.7
    mask_gamma_weight = 0


class PolyLRParams(DictClass):
    """
    Poly Learning Rate Schedules Params.
    """
    power = 0.9
    total_iters = 10000


class OptimizerParams(DictClass):
    """
    Optimizer torch.optim.Adam params.
    """
    lr = 5e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.
    amsgrad = False


class LightningTrainerParms(DictClass):
    """
    Lightning Trainer Params.
    """
    accelerator = 'auto'
    strategy = 'auto'
    devices = -1
    max_epochs = 1000
    num_sanity_val_steps = 0
    precision = '32'
    log_every_n_steps = 1
    accumulate_grad_batches = 1
    enable_progress_bar = True
    enable_model_summary = False
    gradient_clip_val = None
    gradient_clip_algorithm = 'norm'
    deterministic = False


# [DEFAULT CONFIGS]
class DataPathLoaderCall(DictClass):
    """
    DataPathLoader __call__() keyword arguments
    """
    find_masks = Mask.get('mask')
    map_classes = None  # BaseClassMap.to_dict()
    pair_masks = True*Mask.get('mask')


class TorchDatasetInit(DictClass):
    """
    TorchDataset __init__() keyword arguments
    """
    resize_dim = (512, 512)
    output_dim = (256, 256)
    rotation_degrees = 45
    range_brightness = (0.25, 5)
    range_contrast = (0.25, 5)
    range_saturation = (0.25, 5)
    horizontal_flip_prob = 0.5
    vertical_flip_prob = 0.5


class EFSNetInit(DictClass):
    """
    EFSNet __init__() keyword arguments
    """
    verbose = Verbosity.get('verbose')
    input_shape = (1, *TorchDatasetInit.get('output_dim'))
    num_classes = len((ClassValues.to_dict()
                       if DataPathLoaderCall.get('map_classes') is None else
                       MappedClassValues.to_dict()))
    out_channels = 128
    dr_rate = 0.2
    groups = 2
    bias = False
    num_factorized_blocks = 4
    num_super_sdc_blocks = 2
    num_sdc_per_supersdc = 4
    device = Device.get('device')


class LightningModelInit(DictClass):
    """
    LightiningModel __init__() keyword arguments
    """
    model_name = 'EFSNet'
    description = 'EFSNet base'
    metrics = {
        # 'accuracy': accuracy,
        # 'jaccard': jaccard_index
    }
    optimizer = torch.optim.Adam
    optimizer_params = OptimizerParams.to_dict()
    scheduler_params = PolyLRParams.to_dict()
    monitor = 'val_loss'
    frequency = 1
    pos_weight = 5
    ignore_index = -100
    class_weights = [1., 3., 3.]
    save_hyperparameters = True
    resume_training = False
    model_kwargs = EFSNetInit.to_dict()


class TrainerInit(DictClass):
    """
    Trainer __init__() keyword arguments
    """
    use_modelcheckpoint = True
    logger = True
    lightning_trainer_params = LightningTrainerParms.to_dict()
    model_name = LightningModelInit.get('model_name')


class TrainerCall(DictClass):
    """
    Trainer __call__() keyword arguments
    """
    verbose = Verbosity.get('verbose')
    summary_depth = 3
    model_kwargs = LightningModelInit.to_dict()
    batch_size = 32
    resume_training = False


class BaseUpdateLayout(DictClass):
    """
    Configuration for base_update_layout
    """
    title = dict(text='figure',
                 font=dict(family='arial',
                           size=14),
                 x=0.5, y=0.96,
                 xref='paper', yref='container'
                 )
    paper_bgcolor = 'white'
    plot_bgcolor = 'white'
    margin = dict(t=30,
                  b=3,
                  r=3,
                  l=3)
    xaxis = dict(title=dict(text='xaxis',
                            font=dict(
                                family='arial',
                                size=12)),
                 showgrid=False
                 )
    yaxis = dict(title=dict(text='yaxis',
                            font=dict(
                                family='arial',
                                size=12)
                            ),
                 showgrid=False
                 )
    format = 'png'
