from tqdm import tqdm
import visualization_utils as visualization
import torch
import os
import collections

from occFacto.config.config import get_cfg, init_cfg
from occFacto.utils.registry import build_from_cfg, MODELS, DATASETS
from occFacto.models.occFactoModel import OccFacto

SAVE_DIR = "./all_test"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Tokens for target chairs we want
chair_a = "def03f645b3fbd665bb93149cc0adf0"     # Avocado Chair
chair_b = "a9a1147eae9936f76f1e07a56c129dfc"    # Square chair

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

diffFacto.load_state_dict(new_checkpoint)
diffFacto = diffFacto.encoder.to("cuda")
diffFacto.eval()

# Get the validation dataset point clouds
validation_dataset, validation_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

for pcds in tqdm(validation_dataset):
    tokens = pcds['token']

    for i, token in enumerate(tokens):

        if token == chair_a or token == chair_b:
            print(i, tokens)
            # Pass through diffFacto encoder to extract latents
            latents = diffFacto(pcds, device="cuda")
            print(latents[0][i], latents[1][i])
            torch.save({'part': latents[0][i], 'tran': latents[1][i]}, f'{token}.pt')


