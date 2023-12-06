import torch 
from torch import nn
import numpy as np
from occFacto.utils.misc import fps
from occFacto.datasets.evaluation_utils import compute_all_metrics
from tqdm import tqdm
from occFacto.config.config import get_cfg, save_cfg, save_args, init_cfg
from occFacto.utils.registry import build_from_cfg, MODELS, DATASETS, OPTIMS, SCHEDULERS, HOOKS
from occFacto.utils.misc import check_file, build_file, search_ckpt, check_interval, sync, parse_losses
import pickle
import os
import time
import datetime
from occFacto.utils import misc
from occFacto.utils import dist_utils
from einops import rearrange
from occFacto.models.occupancy import OccupancyNetwork
from occFacto.models.occ_types import Options_Occ
import json
import torch.optim as optim
from torch.optim import lr_scheduler

from occFacto.models.occFactoModel import OccFacto
from occFacto.eval.trainerOcc import Trainer
from occFacto.runner.runner import Runner

import collections

# Get diffFacto encoder to generate latents
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
diffFacto = build_from_cfg(cfg.model,MODELS)
checkpoint = torch.load('/home/cs236finalproject/diffFactoCS236/diffFacto/pretrained/chair.pth')['model']

# Parse the state dict keys 
new_checkpoint = collections.OrderedDict()
for key in checkpoint:
    n_key = key
    if key[0:7] == 'module.':
        n_key = key[7:]
    if 'diffusion' in key:
        continue
    new_checkpoint[n_key] = checkpoint[key]
    
# Get the diffFacto encoder
diffFacto.load_state_dict(new_checkpoint)
diffFacto = diffFacto.flow #.to("cuda")
diffFacto.eval()

print(diffFacto)
