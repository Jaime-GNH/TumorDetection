import os
import cv2
import torch
from datetime import datetime
import torch_geometric as tg
from torchmetrics.functional import accuracy, jaccard_index

from TumorDetection.Utils.WorkingDir import WorkingDir


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


class Train(DictClass):
    train = True


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


class ViewerClsParams(DictClass):
    win_title = 'Viewer'
    mask_colormap = cv2.COLORMAP_RAINBOW
    mask_alpha_weight = 0.3
    mask_beta_weight = 0.7
    mask_gamma_weight = 0


class ReadingModes(DictClass):
    gray = cv2.IMREAD_GRAYSCALE
    color = cv2.IMREAD_COLOR
    unchanged = cv2.IMREAD_UNCHANGED


class EarlyStoppingParams(DictClass):
    monitor = 'val_loss'
    mode = 'min'
    patience = 15
    min_delta = 1e-5
    verbose = Verbosity.get('verbose') > 0


class ReduceLROnPLateauParams(DictClass):
    monitor = 'val_loss'
    frequency = 1
    mode = 'min'
    patience = 3
    threshold = 1e-4
    factor = 1e-1
    verbose = Verbosity.get('verbose') > 0
    min_lr = 1e-7


class OptimizerParams(DictClass):
    lr = 1e-3
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
    precision = '16-mixed'
    log_every_n_steps = 1
    accumulate_grad_batches = 1
    enable_progress_bar = True
    enable_model_summary = False
    gradient_clip_val = None
    gradient_clip_algorithm = 'norm'
    deterministic = False


class ModelCkptDir(DictClass):
    cw = WorkingDir.set_wd()
    ckpt_dir = [
        os.path.join(dir_path, 'reporting/ckpt')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


# [DEFAULT CONFIGS]
class DataPathLoaderCall(DictClass):
    find_masks = Mask.get('mask')
    map_classes = BaseClassMap.to_dict()  # {'bening': 'tumor','malignant': 'tumor', 'normal': 'normal'} # None
    pair_masks = True*Mask.get('mask')


class ImageLoaderCall(DictClass):
    read_mode = 'gray'
    class_values = (ClassValues.to_dict()
                    if DataPathLoaderCall.get('map_classes') is None else
                    MappedClassValues.to_dict())


class PreprocessorCall(DictClass):
    resize = False
    invert_grayscale = True
    adjust_exposure = True
    apply_clahe = False
    apply_threshold = False
    detect_contours = False
    clahe_over_last = True
    resize_dim = (512, 512)
    interpolation_method = cv2.INTER_AREA
    img_thresholds_std = [0, 1/4, 1/2, 1]
    clip_hist_percent = 25
    clip_limit = 5.0
    mode = 'last'  # stacked


class ImageToGraphCall(DictClass):
    images_tup_idx = 2
    mask_tup_idx = 3
    dilations = 1
    mask = True
    hypergraph = True
    hypernode_patch_dim = 128
    kernel_kind = 'corner'
    device = Device.get('device')


class GraphDatasetInit(DictClass):
    train_test_split = 0.129  # 780 / 100 -> 100 test samples.
    datapathloader_transforms = DataPathLoaderCall.to_dict()
    imageloader_transforms = ImageLoaderCall.to_dict()
    preprocessor_transforms = PreprocessorCall.to_dict()
    image2graph_transforms = ImageToGraphCall.to_dict()
    patch_dim = None
    seed = 1234567890


class ImageDatasetInit(DictClass):
    train_test_split = 0.129  # 780 / 100 -> 100 test samples.
    datapathloader_transforms = DataPathLoaderCall.to_dict()
    imageloader_transforms = ImageLoaderCall.to_dict()
    preprocessor_transforms = PreprocessorCall.to_dict()
    crop_dim = (256, 256)
    seed = 1234567890


class GraphDataLoaderInit(DictClass):
    graph_dataset_kwargs = GraphDatasetInit.to_dict()
    batch_size = 128
    num_workers = 0
    drop_last = True


class ImageDataLoaderInit(DictClass):
    image_dataset_kwargs = ImageDatasetInit.to_dict()
    batch_size = 64
    drop_last = True
    num_workers = 0  # os.cpu_count()//2
    pin_memory = False  # Device.get('device') != 'cpu'
    persistent_workers = False  # True


class ConvLayerKwargs(DictClass):
    dropout = 0.2
    act = 'relu'
    act_first = False
    act_kwargs = None
    norm = None
    norm_kwargs = None
    num_layers = 1
    jk = None


class BaseGNNInit(DictClass):
    h_size = [64, 32, 16, 8]
    conv_layer_type = tg.nn.GraphSAGE
    conv_layer_kwargs = ConvLayerKwargs.to_dict()


class ImageGNNInit(DictClass):
    h_size = [256, 128, 64, 32]
    conv_layer_type = tg.nn.GraphSAGE
    conv_layer_kwargs = ConvLayerKwargs.to_dict()
    hypernode_patch_dims = [64, 16, 4, 1]
    kernel_kind = 'corner'
    last_kernel = 'star'


class LightningModelInit(DictClass):
    model_name = 'model_' + 'GraphSAGE-patch(64)-hyperpixel-iterative_multiloss'  # + datetime.now().strftime('%d%m%Y_%H%M')
    description = 'SAGE Conv con batch de 128, hypernode_patch_dim de 64 y una pérdida por nivel.'
    criterion = len(ImageGNNInit.get('hypernode_patch_dims'))*[torch.nn.BCEWithLogitsLoss(reduction='mean')]
    metrics = {
        'accuracy': accuracy,
        'jaccard': jaccard_index
    }
    optimizer = torch.optim.Adam
    save_hyperparameters = True
    resume_training = False
    use_reducelronplateau = True
    rlr_kwargs = ReduceLROnPLateauParams.to_dict()
    gnn_kwargs = ImageGNNInit.to_dict()


class TrainerInit(DictClass):
    use_earlystopping = True
    use_modelcheckpoint = True
    es_kwargs = EarlyStoppingParams.to_dict()
    logger = True
    lightning_trainer_params = LightningTrainerParms.to_dict()
    model_name = LightningModelInit.get('model_name')


class TrainerCall(DictClass):
    model_kwargs = LightningModelInit.to_dict()
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
