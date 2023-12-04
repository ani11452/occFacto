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
import mcubes

# Get diffFacto encoder to generate latents
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
diffFacto = build_from_cfg(cfg.model,MODELS)
diffFacto = diffFacto.encoder.to("cuda")

# Initialize the the Trainer class
trainer = Trainer()

# Intialize the Model we want to Eval
model = OccFacto()
checkpoint = torch.load("occFactoDiffFreezeTraining2/occFacto_best_model.pth")
model.load_state_dict(checkpoint)
model.to("cuda")
val_dataset, val_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

# Generate eval cube:
# eval_p = 1.1 * trainer.make_3d_grid((-0.5,)*3, (0.5,)*3, (25,)*3)

i = 5
for exam in val_dataset:
    if i <= 0:
        break
    latents = diffFacto(exam, device="cuda")
    latents = torch.cat(tuple(latents), dim=1)

    print(latents.shape)

    print(trainer.eval_metrics(model, latents, exam))

    # Model Eval
    # occ_hats = trainer.eval_points(model, latents)

    # # Reshape
    # occ_hats = occ_hats.view(25, 25, 25)
    # occ_hats = np.pad(occ_hats, 1, 'constant', constant_values=-1e6)
    # occ_hats = mcubes.smooth(occ_hats)

    # # Apply marching cubes algorithm
    # vertices, triangles = mcubes.marching_cubes(occ_hats, 0)
    # mcubes.export_mesh(vertices, triangles, "occFactoDiffFreezeTraining2/best_model_pred_" + str(i) + ".dae")

    
    # data = {
    #     "val_example": exam,
    #     "predictions": occ_hats.cpu()
    # }

    # file_path = "occFactoDiffFreezeTraining2/best_model_pred_" + str(i) + ".pkl"
    
    # # Save the dictionary to a binary file using pickle
    # with open(file_path, 'wb') as pickle_file:
    #     pickle.dump(data, pickle_file)
    
    
    i -= 1



