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

# Get diffFacto
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
model = build_from_cfg(cfg.model,MODELS)
diffFacto = model.encoder


class OccFacto(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize diffFacto
        self.diffFacto = model.encoder

        # Linear Projection Layer:
        self.Linear = nn.Linear(262, 128, bias=False)

        # Intialize Spaghetti
        self.opt = Options_Occ()
        self.spagOcc = OccupancyNetwork(self.opt)

    def getLatents(self, pcds):
        # Gets latents for the given ShapeNet examples
        out = self.diffFacto(pcds, device="cuda")
        out = torch.cat(tuple(out), dim=1)
        return self.Linear(out.permute(0, 2, 1))

    def forward(self, pcds, x):
        # Predicts occupancies for sampled points from ShapeNet examples
        latents = self.getLatents(pcds)
        occs = self.spagOcc(x, latents)
        return occs









    