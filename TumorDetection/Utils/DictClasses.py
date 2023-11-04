import os
import cv2
import torch
from datetime import datetime
import multiprocessing as mp
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
    dir_path = [
        os.path.join(dir_path, 'Dataset_BUSI_with_GT')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'Dataset_BUSI_with_GT' in dir_name
    ][0]


class ResourcesPathDir(DictClass):
    dir_path = [
        os.path.join(dir_path, 'resources')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'resources' in dir_name
    ][0]


class ReportingPathDir(DictClass):
    dir_path = [
        os.path.join(dir_path, 'reporting')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'reporting' in dir_name
    ][0]


class PicklePathDir(DictClass):
    dir_path = [
        os.path.join(dir_path, 'pickled_objects')
        for dir_path, dir_name, _ in os.walk(WorkingDir.getwd_from_path(os.getcwd()))
        if 'pickled_objects' in dir_name
    ][0]


class Files(DictClass):
    pickle_loader = 'BUSI_Graphs_stacked_3classes_780_1-29-227.pkl'
    # BUSI_Graphs_{preproceso_mode}_{num_classes}classes_{len}.pkl


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
    patience = 7
    min_delta = 1e-5


class ReduceLROnPLateauParams(DictClass):
    monitor = 'val_loss'
    mode = 'min'
    patience = 3
    min_delta = 1e-4
    factor = 1e-1
    verbose = Verbosity.get('verbose') > 1
    min_lr = 1e-7


class OptimizerParams(DictClass):
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.
    amsgrad = False


class ModelCkptDir(DictClass):
    ckpt_dir = [
        os.path.join(dir_path, 'reporting')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'reporting' in dir_name
    ]


# [DEFAULT CONFIGS]
class DataPathLoaderCall(DictClass):
    find_masks = Mask.get('mask')
    map_classes = None  # BaseClassMap.to_dict() {'bening': 'tumor','malignant': 'tumor', 'normal': normal}
    pair_masks = True*Mask.get('mask')


class ImageLoaderCall(DictClass):
    read_mode = 'gray'
    class_values = ClassValues.to_dict()


class PreprocessorCall(DictClass):
    invert_grayscale = True
    adjust_exposure = True
    apply_clahe = True
    apply_threshold = False
    detect_contours = False
    clahe_over_last = True
    img_thresholds_std = [0, 1/4, 1/2, 1]
    clip_hist_percent = 25
    clip_limit = 5.0
    mode = 'stacked'


class ImageToGraphCall(DictClass):
    images_tup_idx = 2
    mask_tup_idx = 3
    dilations = (1, 29, 227)
    mask = True
    kernel_kind = 'hex'
    device = Device.get('device')


class GraphDatasetInit(DictClass):
    train = Train.get('train')
    train_test_split = 0.129  # 780 / 100 -> 100 test samples.
    inference = False
    datapathloader_transforms = DataPathLoaderCall.to_dict()
    imageloader_transforms = ImageLoaderCall.to_dict()
    preprocessor_transforms = PreprocessorCall.to_dict()
    image2graph_transforms = ImageToGraphCall.to_dict()


class GraphDataLoaderCall(DictClass):
    batch_size = 16
    num_workers = 0
    shuffle = True
    drop_last = True


class HyperGNNInit(DictClass):
    h_size = [32, 64, 128]
    layer_type = tg.nn.GATConv


class LightningModelInit(DictClass):
    model_name = 'model_' + datetime.now().strftime('%d%m%Y_%H%M')
    criterion = torch.nn.functional.cross_entropy
    metrics = {
        'accuracy': accuracy,
        'jaccard': jaccard_index
    }
    optimizer = torch.optim.Adam
    save_hyperparameters = True
    resume_training = False
    use_earlystopping = True
    use_modelcheckpoint = True
    use_reducelronplateau = True
    es_kwargs = EarlyStoppingParams.to_dict()
    rlr_kwargs = ReduceLROnPLateauParams.to_dict()
    model_kwargs = HyperGNNInit.to_dict()


class TrainerInit(DictClass):
    accelerator = 'auto'
    strategy = 'auto'
    devices = -1
    precision = '32-true'
    logger = True
    max_epochs = 1000
    val_check_interval = 0.
    check_val_every_n_epoch = 1
    enable_progress_bar = True
    enable_model_summary = False
    accumulate_grad_batches = 1
    gradient_clip_val = None
    gradient_clip_algorithm = 'norm'
    deterministic = False


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
