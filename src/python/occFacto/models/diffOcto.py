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

init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')

cfg = get_cfg()
model = build_from_cfg(cfg.model,MODELS)
model.to("cuda")
model = model.encoder
model.eval()

train_dataset, train_sampler = build_from_cfg(cfg.dataset.train, DATASETS, distributed=False)

for batch_idx, pcds in enumerate(train_dataset):
    result = model(pcds, device="cuda")
    print(result)
    break