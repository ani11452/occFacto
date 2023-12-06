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
    e = torch.randn(256)
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

def swap_latent(part_a, part_b, i):

    # Assume we have part latents_a and latents_b
    # 256 x 4
    
    # Add dim
    part_a = part_a.unsqueeze(0).detach()
    part_b = part_b.unsqueeze(0).detach()

    # Swap
    part_b_i = part_b[:, :, i]
    part_b[:, :, i] = part_a[:, :, i]
    part_a[:, :, i] = part_b_i

    # Generate new Transformations
    tran_a = sample_transformation(part_a)
    tran_b = sample_transformation(part_b)

    res = {
        'Shape_A' : torch.cat((part_a, tran_a), dim=1),
        'Shape_B' : torch.cat((part_b, tran_b), dim=1)
    }

    return res


def interpolate_latent(part_a, part_b, i, ms):

    # Add dim
    part_a = part_a.unsqueeze(0).detach()
    part_b = part_b.unsqueeze(0).detach()

    new = torch.zeros_like(part_a) + part_a

    print("Squeezed Parts", part_a, part_b)

    # Get latent
    part_a_i = part_a[:, :, i].squeeze()
    part_b_i = part_b[:, :, i].squeeze()

    print("Single Parts", part_a_i, part_b_i)


    # Interpolation
    values = []
    for alpha in ms:
        # print(part_a_i, part_b_i)
        lat = alpha * part_a_i + (1 - alpha) * part_b_i
        print("Lat", lat)
        new[:, :, i] = lat
        tran_a = sample_transformation(new)

        values.append(torch.cat((new, tran_a), dim=1))

    # # Set up return value
    # keys = []
    # for n in range(1, m + 1):
    #     keys.append(str(n))

    # Create value
    # interpolations = zip(keys, values)
    # interpolations = dict(interpolations)

    # return interpolations
    return values


def keepSomeChangeSome(pcds, replace, times): 
    # Sample Actual latents
    parts, _ = diffFacto(pcds, device="cuda")

    # Zero out the values we want to replace
    replace = torch.tensor([(x in replace) for x in range(4)])
    parts[0, :, replace] = 0

    # Resample and save new ones
    values = []
    for i in range(times):
        new_parts, _ = diffFacto(pcds, device="cuda")
        part = parts + new_parts[0, :, replace]
        tran = sample_transformation(part)
        values.append(torch.cat((part, tran), dim=1))
        
    return values





SAVE_DIR = "./all_test_inter0"

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    CHAIR_A = "def03f645b3fbd665bb93149cc0adf0"     # Avocado Chair
    CHAIR_B = "a1213da0e7efffcafebad4f49b26ec52"    # Square chair

    occFacto = OccFacto()
    occFacto.to("cuda")

    occFacto.load_state_dict(torch.load('/home/cs236finalproject/diffFactoCS236/src/python/occFacto/models/occFactoDiffFreezeTrainingLegitFrozenDiffFactoReg3/occFacto_best_model_loss.pth')['model_state_dict'])
    occFacto.eval()

    latents_a = torch.load(f'{CHAIR_A}.pt')
    latents_b = torch.load(f'{CHAIR_B}.pt')

    part_latents_a = latents_a['part']
    part_latents_b = latents_b['part']

    print(part_latents_a, part_latents_b)

    # swap0 = swap_latent(part_latents_a, part_latents_b, 2)

    # part_latents_a = swap0['Shape_A']
    # part_latents_b = swap0['Shape_B']

    # mesher = visualization.Mesher(occFacto, SAVE_DIR, tag="swap")
    # mesher.generate_mesh(part_latents_a, CHAIR_A)
    # mesher.generate_mesh(part_latents_b, CHAIR_B)

    # viz_a = visualization.Visualizer(CHAIR_A, tag="swap")
    # viz_a.save_view()

    # viz_b = visualization.Visualizer(CHAIR_B, tag="swap")
    # viz_b.save_view()

    interpolate2 = interpolate_latent(part_latents_a, part_latents_b, 1, [0.0, 0.25, 0.50, 0.75, 1.0])

    # print(interpolate2)


    char = 'a'
    for lat in interpolate2:
        mesher = visualization.Mesher(occFacto, SAVE_DIR, tag="inter1")
        mesher.generate_mesh(lat, char)

        char = ord(char) + 1
        char = chr(char)
        print(char)

    vizA = visualization.Visualizer('a', tag="inter1")
    vizA.save_view()

    vizB = visualization.Visualizer('b', tag="inter1")
    vizB.save_view()

    vizC = visualization.Visualizer('c', tag="inter1")
    vizC.save_view()

    vizD = visualization.Visualizer('d', tag="inter1")
    vizD.save_view()

    vizE = visualization.Visualizer('e', tag="inter1")
    vizE.save_view()

    