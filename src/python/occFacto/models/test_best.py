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

import pickle

# Get diffFacto encoder to generate latents
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
diffFacto = build_from_cfg(cfg.model,MODELS)
diffFacto = diffFacto.encoder.to("cuda")

# Initialize the the Trainer class

'''
train_params = {
    "print_every": 1,
    "visualize_every": 1, 
    "checkpoint_every": 1,
    "backup_every": 1,
    "validate_every": 1}
'''
# trainer = Trainer()

# Intialize the Model we want to Train
model = OccFacto()
checkpoint = torch.load("occFactoDiffFreezeTraining/occFacto_best_model_epoch_149.pth")
model.load_state_dict(checkpoint)
model.to("cuda")
val_dataset, val_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

i = 5
for exam in val_dataset:
    if i <= 0:
        break
    latents = diffFacto(exam, device="cuda")
    latents = torch.cat(tuple(latents), dim=1)

    # Model outputs
    occPoints = exam["occs"][0].to("cuda")
    occPreds = model(latents, occPoints)


    data = {
            "val_example": exam,
            "predictions": occPreds}

    file_path = "best_model_pred_" + str(i) + ".pkl"
    
    # Save the dictionary to a binary file using pickle
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    
    i -= 1




