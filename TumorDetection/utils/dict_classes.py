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
    verbose = 3


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
