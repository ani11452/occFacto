from tqdm import tqdm
import visualization_utils as visualization
import torch
import os
import collections

from occFacto.config.config import get_cfg, init_cfg
from occFacto.utils.registry import build_from_cfg, MODELS, DATASETS
from occFacto.models.occFactoModel import OccFacto
from occFacto.eval.trainerOcc import Trainer


SAVE_DIR = "./all_test_best"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Tokens for target chairs we want
# chair_a = "def03f645b3fbd665bb93149cc0adf0"     # Avocado Chair
# chair_b = "a9a1147eae9936f76f1e07a56c129dfc"    # Square chair
chair_a = "5402eecc67e489502fa77440dcb93214"
chair_b = "5402eecc67e489502fa77440dcb93214"

# Get diffFacto encoder to generate latents
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
diffFacto = build_from_cfg(cfg.model, MODELS)
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
# diffFacto.load_state_dict(new_checkpoint)
# diffFacto = diffFacto.encoder.to("cuda")
# diffFacto.eval()

# Intialize the occFacto Model
occFacto = OccFacto()
occFacto.to("cuda")

occFacto.load_state_dict(torch.load('/home/cs236finalproject/diffFactoCS236/src/python/occFacto/models/occFactoDiffFreezeTrainingLegitFrozenDiffFactoReg/occFacto_best_model_iou.pth')['model_state_dict'])
occFacto.eval()  # Set the model to evaluation mode

# Get the validation dataset point clouds
validation_dataset, validation_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

latents_a = torch.load(f'{chair_a}.pt')
latents_b = torch.load(f'{chair_b}.pt')

latents_a = (latents_a['part'].unsqueeze(0), latents_a['tran'].unsqueeze(0))
latents_b = (latents_b['part'].unsqueeze(0), latents_b['tran'].unsqueeze(0))


latent_a = torch.cat(latents_a, dim=1)
latent_b = torch.cat(latents_b, dim=1)

print(latent_a.size(), latent_b.size())

mesher = visualization.Mesher(occFacto, SAVE_DIR)
mesher.generate_mesh(latent_a, chair_a)
mesher.generate_mesh(latent_b, chair_b)

viz_a = visualization.Visualizer(chair_a)
viz_a.save_view()

viz_b = visualization.Visualizer(chair_b)
viz_b.save_view()

