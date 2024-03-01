import os
os.environ['KERAS_BACKEND'] = 'torch'
import cv2
import torch
from keras.optimizers import Adam, schedules as ksh
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy, MeanIoU
import torch_geometric as tg
from torchmetrics.functional import accuracy, jaccard_index

from TumorDetection.utils.working_dir import WorkingDir


class DictClass:

    @classmethod
    def to_dict(cls):
        """

        :return:
        """
        return {k: v for k, v in vars(cls).items() if not k.startswith('__')}

    @classmethod
    def get(cls, name):
        """

        :return:
        """
        return vars(cls)[name]


# [MAPS]
class ClassValues(DictClass):
    normal = 0
    benign = 1
    malignant = 2


class MappedClassValues(DictClass):
    normal = 0
    tumor = 1


class BaseClassMap(DictClass):
    normal = 'normal'
    benign = 'tumor'
    malignant = 'tumor'


# [PARAMS]
class Verbosity(DictClass):
    verbose = 1


class Mask(DictClass):
    mask = True


class Device(DictClass):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataPathDir(DictClass):
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'Dataset_BUSI_with_GT')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'Dataset_BUSI_with_GT' in dir_name
    ][0]
    os.chdir(cw)


class ResourcesPathDir(DictClass):
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'resources')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'resources' in dir_name
    ][0]
    os.chdir(cw)


class ReportingPathDir(DictClass):
    cw = WorkingDir.set_wd()
    dir_path = [
        os.path.join(dir_path, 'reporting')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


class ModelCkptDir(DictClass):
    cw = WorkingDir.set_wd()
    ckpt_dir = [
        os.path.join(dir_path, 'reporting/ckpt')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


class ViewerClsParams(DictClass):
    win_title = 'Viewer'
    mask_colormap = cv2.COLORMAP_RAINBOW
    mask_alpha_weight = 0.3
    mask_beta_weight = 0.7
    mask_gamma_weight = 0


class ModelCheckpointParams(DictClass):
    save_best_only = True
    save_weights_only = True
    verbose = 1 if Verbosity.get('verbose') > 0 else 0


class CompileParams(DictClass):
    optimizer = Adam(
        learning_rate=ksh.PolynomialDecay(
            initial_learning_rate=5e-4,
            end_learning_rate=1e-7,
            decay_steps=2000,
            cycle=True
        ),
        weight_decay=0.0004,
        ema_momentum=0.9
    )
    class_weights = [1., 3, 3]
    ignore_index = -100
    pos_weight = 5


class PolyLRParams(DictClass):
    power = 0.9
    total_iters = 10


class OptimizerParams(DictClass):
    lr = 5e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.
    amsgrad = False


class LightningTrainerParms(DictClass):
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
    find_masks = Mask.get('mask')
    map_classes = None  # BaseClassMap.to_dict()
    pair_masks = True*Mask.get('mask')


class ImageLoaderCall(DictClass):
    read_mode = 'gray'
    class_values = (ClassValues.to_dict()
                    if DataPathLoaderCall.get('map_classes') is None else
                    MappedClassValues.to_dict())


class PatchDatasetCall(DictClass):
    normalize = True
    resize = True
    resize_dim = (512, 512)
    interpolation_method = cv2.INTER_AREA
    patch_dim = (256, 256)
    patch_step = 64
    test_size = 0.1
    mode = 'patches'
    shuffle = True
    random_state = 1234567890
    filter_empty = True


class TorchDatasetInit(DictClass):
    resize_dim = (512, 512)
    output_dim = (256, 256)
    rotation_degrees = 45
    max_brightness = 5
    max_contrast = 5
    max_saturation = 5
    horizontal_flip_prob = 0.5
    vertical_flip_prob = 0.5


class EfficientNetInit(DictClass):
    groups = 2
    dr_rate = 0.2
    num_factorized_blocks = 4
    num_super_sdc_blocks = 2
    max_filters = 128
    activation_layer = 'prelu'
    kernel_initializer = 'he_uniform'
    use_bias = False


class EFSNetInit(DictClass):
    verbose = Verbosity.get('verbose')
    input_shape = (1, *TorchDatasetInit.get('output_dim'))
    num_classes = len(ImageLoaderCall.get('class_values'))
    filters = 128
    dr_rate = 0.2
    groups = 2
    bias = False
    num_factorized_blocks = 4
    num_super_sdc_blocks = 2
    num_sdc_per_supersdc = 4
    # pos_weight = 5
    # ignore_index = -100
    # class_weights = [1., 3., 3.]
    # EfficientNetInit = EfficientNetInit.to_dict()
    device = Device.get('device')
    name = 'EFSNet'


class LightningModelInit(DictClass):
    model_name = 'EFSNet'
    description = 'EFSNet base'
    metrics = {
        'accuracy': accuracy,
        'jaccard': jaccard_index
    }
    optimizer = torch.optim.Adam
    monitor = 'val_loss'
    frequency = 1
    pos_weight = 5
    ignore_index = -100
    class_weights = [1., 3., 3.]
    save_hyperparameters = True
    resume_training = False
    model_kwargs = EFSNetInit.to_dict()


class TrainerInit(DictClass):
    use_modelcheckpoint = True
    logger = True
    lightning_trainer_params = LightningTrainerParms.to_dict()
    model_name = LightningModelInit.get('model_name')


class TrainerCall(DictClass):
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
