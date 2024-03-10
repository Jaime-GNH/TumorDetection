import cv2


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


# [PARAMS]
class Verbosity(DictClass):
    verbose = 1


class ReadingModes(DictClass):
    gray = cv2.IMREAD_GRAYSCALE
    color = cv2.IMREAD_COLOR
    unchanged = cv2.IMREAD_UNCHANGED


class EarlyStoppingParams(DictClass):
    monitor = 'val_loss'
    mode = 'min'
    patience = 50
    min_delta = 1e-7
    verbose = 1 if Verbosity.get('verbose') > 0 else 0
    restore_best_weights = True


class ReduceLROnPLateauParams(DictClass):
    monitor = 'val_loss'
    frequency = 1
    mode = 'min'
    patience = 3
    threshold = 1e-4
    factor = 1e-1
    verbose = 1 if Verbosity.get('verbose') > 0 else 0
    min_lr = 1e-7


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


class ModelCheckpointParams(DictClass):
    save_best_only = True
    save_weights_only = True
    verbose = 1 if Verbosity.get('verbose') > 0 else 0


class TrainParams(DictClass):
    epochs = 1000
    batch_size = 32
    precision = 'mixed_float16'
    shuffle = True
    use_reducelronplateau = False
    use_earlystopping = False
    use_modelcheckpoint = True
    log_tensorboard = False
    rlr_kwargs = ReduceLROnPLateauParams.to_dict()
    es_kwargs = EarlyStoppingParams.to_dict()
    ckpt_params = ModelCheckpointParams.to_dict()


class DropoutPathParams(DictClass):
    p = 0.2
    walks_per_node = 1
    walk_length = 4
    num_nodes = None
    is_sorted = False


class DropoutEdgeParams(DictClass):
    p = 0.3
    force_undirected = False


class ModelCkptDir(DictClass):
    cw = WorkingDir.set_wd()
    ckpt_dir = [
        os.path.join(dir_path, 'reporting/ckpt')
        for dir_path, dir_name, _ in os.walk(os.getcwd())
        if 'reporting' in dir_name
    ][0]
    os.chdir(cw)


class ImageLoaderCall(DictClass):
    """
    ImageLoader __call__() keyweord arguments
    """
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
    dilations = (1, 4, 16, 64)
    mask = True
    hypergraph = False
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
    batch_size = 32
    drop_last = True
    num_workers = 0  # os.cpu_count()//2
    pin_memory = False  # Device.get('device') != 'cpu'
    persistent_workers = False  # True


class ConvLayerParams(DictClass):
    dropout = 0.2
    act = 'relu'
    act_first = False
    act_kwargs = None
    norm = None
    norm_kwargs = None
    num_layers = 1
    jk = None


class BaseGNNInit(DictClass):
    h_size = [16, 32, 64, 128]
    num_classes = len(ImageLoaderCall.get('class_values'))
    conv_layer_type = tg.nn.GraphSAGE
    use_dropoutpath = True
    use_dropoutedge = True
    dilations = (1, 4, 16, 64)
    dropoutpath_params = DropoutPathParams.to_dict()
    dropoutedge_params = DropoutEdgeParams.to_dict()
    conv_layer_kwargs = ConvLayerParams.to_dict()


class ImageGNNInit(DictClass):
    h_size = [256, 128, 64, 32]
    conv_layer_type = tg.nn.GraphSAGE
    conv_layer_kwargs = ConvLayerParams.to_dict()
    hypernode_patch_dims = [64, 16, 4, 1]
    kernel_kind = 'corner'
    last_kernel = 'star'


class EfficientNetInit(DictClass):
    groups = 2
    dr_rate = 0.2
    num_factorized_blocks = 4
    num_super_sdc_blocks = 2
    max_filters = 128
    activation_layer = 'prelu'
    kernel_initializer = 'he_uniform'
    use_bias = False


class EFSNetModelInit(DictClass):
    verbose = Verbosity.get('verbose')
    input_shape = (*TorchDatasetInit.get('output_dim'), 1)
    num_classes = len(ImageLoaderCall.get('class_values'))
    EfficientNetInit = EfficientNetInit.to_dict()
    compile_params = CompileParams.to_dict()
    train_params = TrainParams.to_dict()
    model_name = 'EFSNet_torch'


class LightningModelInit(DictClass):
    model_name = 'model_' + 'GraphSAGE-droplinking'  # + datetime.now().strftime('%d%m%Y_%H%M')
    description = 'SAGE Conv con eliminación aleatoria de enlaces en el entrenamiento.'
    # criterion = len(ImageGNNInit.get('hypernode_patch_dims'))*[torch.nn.BCEWithLogitsLoss(reduction='mean',
    #                                                                                       pos_weight=torch.tensor(10))]
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    metrics = {
        'accuracy': accuracy,
        'jaccard': jaccard_index
    }
    optimizer = torch.optim.Adam
    save_hyperparameters = True
    resume_training = False
    use_reducelronplateau = True
    rlr_kwargs = ReduceLROnPLateauParams.to_dict()
    gnn_kwargs = BaseGNNInit.to_dict()  # ImageGNNInit.to_dict()


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
