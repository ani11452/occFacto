from tqdm import tqdm
import visualization_utils as visualization
import torch
import os
import collections

from occFacto.config.config import get_cfg, init_cfg
from occFacto.utils.registry import build_from_cfg, MODELS, DATASETS
from occFacto.models.occFactoModel import OccFacto

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
diffFacto.to("cuda")
diffFacto.eval()

# Get Flow Network Part Stylizer Prior
flows = diffFacto.encoder.flow

# Define Function to Get Zs
def sample_part_latent(batch=1):
    e = 2 * torch.rand(256)
    e = e.unsqueeze(0)
    e = e.repeat(batch, 1).to("cuda")

    z = []
    for flow in flows:
        z.append(flow(e))

    z = torch.stack(z)
    z = z.permute(1, 2, 0)
    return z

# Get Transformation Sampler
transformation = diffFacto.encoder.part_aligner

def sample_transformation(zs, batch=1):
    y = torch.randn(batch, 32)
    y = y.repeat(1, 1, 4).to("cuda")
    out = torch.cat(transformation(x=zs, noise=y), dim=1)
    
    return out





SAVE_DIR = "./all_test_generate"

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    occFacto = OccFacto()
    occFacto.to("cuda")

    occFacto.load_state_dict(torch.load('/home/cs236finalproject/diffFactoCS236/src/python/occFacto/models/occFactoDiffFreezeTrainingLegitFrozenDiffFacto/occFacto_best_model_loss.pth')['model_state_dict'])
    occFacto.eval()

    # zs_1 = sample_part_latent()
    # zs_2 = sample_part_latent()
    # zs_3 = sample_part_latent()
    zs_4 = sample_part_latent()

    zs_1 = 5*torch.rand_like(zs_4) - 2.5
    zs_2 = 5*torch.rand_like(zs_4) - 2.5
    zs_3 = 5*torch.rand_like(zs_4) - 2.5
    zs_4 = 5*torch.rand_like(zs_4) - 2.5

    tran_1 = sample_transformation(zs_1)
    tran_2 = sample_transformation(zs_2)
    tran_3 = sample_transformation(zs_3)
    tran_4 = sample_transformation(zs_4)

    z_1 = torch.cat((zs_1, tran_1), dim=1)
    z_2 = torch.cat((zs_2, tran_2), dim=1)
    z_3 = torch.cat((zs_3, tran_3), dim=1)
    z_4 = torch.cat((zs_4, tran_4), dim=1)

    print(z_1.shape)

    mesher = visualization.Mesher(occFacto, SAVE_DIR, tag="generate")
    mesher.generate_mesh(z_1, "gen1")
    mesher.generate_mesh(z_2, "gen2")
    mesher.generate_mesh(z_3, "gen3")
    mesher.generate_mesh(z_4, "gen4")

    viz_a = visualization.Visualizer("gen1", tag="generate")
    viz_a.save_view()

    viz_b = visualization.Visualizer("gen2", tag="generate")
    viz_b.save_view()

    viz_c = visualization.Visualizer("gen3", tag="generate")
    viz_c.save_view()

    viz_d = visualization.Visualizer("gen4", tag="generate")
    viz_d.save_view()

    