import os
import cv2
import torch
from torchmetrics.functional import accuracy, jaccard_index


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


class DataPathDir(DictClass):
    dir_path = [
        os.path.join(dir_path, 'Dataset_BUSI_with_GT')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'Dataset_BUSI_with_GT' in dir_name
    ][0]


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
    find_masks = True
    map_classes = None  # {'bening': 'tumor','malignant': 'tumor', 'normal': normal}
    pair_masks = True


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
    train = True
    train_test_split = 0.2
    deterministic = True
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GNNModelInit(DictClass):
    h_size = [32, 64, 128]


class LightningModelInit(DictClass):
    model_name = 'model'
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
    model_kwargs = GNNModelInit.to_dict()


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
