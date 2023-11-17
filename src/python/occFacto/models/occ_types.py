# FROM SPAGHETTI

from __future__ import annotations
import os
import pickle
import numpy as np
import torch
from typing import Tuple, List, Union, Optional, Callable
from enum import Enum, unique
import torch.optim.optimizer
import torch.utils.data
import sys

IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints/'
CACHE_ROOT = f'{DATA_ROOT}cache/'
UI_OUT = f'{DATA_ROOT}ui_export/'
UI_RESOURCES = f'{DATA_ROOT}/ui_resources/'
# Shapenet_WT = f'{DATA_ROOT}/ShapeNetCore_wt/'
# Shapenet = f'{DATA_ROOT}/ShapeNetCore.v2/'
MAX_VS = 100000
MAX_GAUSIANS = 32

if DEBUG:
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

V_Mesh = Tuple[ARRAY, ARRAY]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, ARRAY, Tuple[int, int, int]]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
Optimizer = torch.optim.Adam
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
Subset = torch.utils.data.Subset

class Options_Occ:

    def load(self):
        device = self.device
        if os.path.isfile(self.save_path):
            print(f'loading opitons from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            options.device = device
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder):
            # self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @property
    def info(self) -> str:
        return f'{self.model_name}_{self.tag}'

    @property
    def cp_folder(self):
        return f'{CHECKPOINTS_ROOT}{self.info}'

    @property
    def save_path(self):
        return f'{CHECKPOINTS_ROOT}{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.device = 'cuda'
        self.tag = 'airplanes'
        self.dataset_name = 'shapenet_airplanes_wm_sphere_sym_train'
        self.epochs = 2000
        self.model_name = 'spaghetti'
        self.dim_z = 128
        self.pos_dim = 128 - 3
        self.dim_h = 128
        self.dim_zh = 128
        self.num_gaussians = 16
        self.min_split = 4
        self.max_split = 12
        self.gmm_weight = 1
        self.decomposition_network = 'transformer'
        self.decomposition_num_layers = 4
        self.num_layers = 4
        self.num_heads = 4
        self.num_layers_head = 6
        self.num_heads_head = 8
        self.head_occ_size = 5
        self.head_occ_type = 'skip'
        self.batch_size = 18
        self.num_samples = 2000
        self.dataset_size = -1
        self.symmetric = (True, False, False)
        self.data_symmetric = (True, False, False)
        self.lr_decay = .9
        self.lr_decay_every = 500
        self.warm_up = 2000
        self.reg_weight = 1e-4
        self.disentanglement = True
        self.use_encoder = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = 0.3
        self.augmentation_scale = .2
        self.augmentation_translation = .3
        self.fill_args(kwargs)