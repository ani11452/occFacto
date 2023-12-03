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
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from itertools import cycle

'''
# Get diffFacto
init_cfg('/home/cs236finalproject/diffFactoCS236/src/config_files/gen_occ.py')
cfg = get_cfg()
model = build_from_cfg(cfg.model,MODELS)
diffFacto = model.encoder.to("cuda")
'''

class OccFacto(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize diffFacto
        # self.diffFacto = model.encoder

        # Linear Projection Layer:
        self.Linear = nn.Linear(262, 128, bias=False)

        # Intialize Spaghetti
        self.opt = Options_Occ()
        self.spagOcc = OccupancyNetwork(self.opt)
    '''
    def getLatents(self, pcds):
        # Gets latents for the given ShapeNet examples
        out = self.diffFacto(pcds, device="cuda")
        out = torch.cat(tuple(out), dim=1)
        return self.Linear(out.permute(0, 2, 1))
    '''
    def forward(self, out, x):
        # Predicts occupancies for sampled points from ShapeNet examples
        # latents = self.getLatents(pcds)
        latents = self.Linear(out.permute(0, 2, 1))
        occs = self.spagOcc(x, latents)
        return occs

'''
def train_loop(train_dataset, train_datasetNeg, optimizer, log):
        occFacto.train()
        
        train_loss = []
        for epoch in tqdm(range(epochs)):
            print(f"Training Epoch {epoch + 1}")
            avg_accuracy = []

            for pcds in tqdm(zip(train_dataset, train_datasetNeg)):
                pcds, opp = pcds
                out = diffFacto(pcds, device="cuda")
                out = torch.cat(tuple(out), dim=1)
                
                if pcds["input"].size(0) != 32:
                    continue

                opp = torch.cat((opp["input"], opp["input"], opp["input"][0:6, :, :]), dim=0)

                # Get eval pts:
                initial_rng_state = torch.random.get_rng_state()
                seed = 42
                torch.manual_seed(seed)

                random_indices = torch.randperm(pcds["input"].size(1))[:1000]
                pos = pcds["input"][:, random_indices, :]
                neg = opp[:, random_indices, :]
                xs = torch.cat((pos, neg), dim=1)
                b, n, _ = xs.size()
                ys = torch.ones(b, n, dtype=torch.float32)
                ys[:, 1000:] = 0
                random_indices = torch.randperm(xs.size(1))
                xs, ys = xs[:, random_indices, :], ys[:, random_indices]
                xs = xs.to("cuda")
                ys = ys.to("cuda")

                torch.random.set_rng_state(initial_rng_state)

                # Model outputs
                out = occFacto(out, xs)

                # Assert
                assert out.size() == ys.size()

                # Loss:
                loss = loss_f(out, ys)
                loss.backward()

                # Optimizer
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

                # Eval
                out = torch.sigmoid(out)
                bin = (out >= 0.5).int()
                correct_predictions = torch.sum(bin == ys)
                total_predictions = bin.numel()
                accuracy = correct_predictions.item() / total_predictions
                avg_accuracy.append(accuracy)
            
            avg_accuracy = torch.mean(torch.tensor(avg_accuracy)).item()
            print(f"Epoch [{epochs + 1}/{epochs}], Training Loss: {loss.item()}, Accuracy: {avg_accuracy}%")
            report = f"Epoch [{epochs + 1}/{epochs}], Training Loss: {loss.item()}, Accuracy: {avg_accuracy}%"
            log.write(report)
            torch.save(occFacto.state_dict(), f"dummy_epoch{epoch}.pth")

        # Save params after training
        torch.save(occFacto.state_dict(), "dummy_final_params.pth")
        return train_loss

if __name__ == '__main__':

    # Load Model
    occFacto = OccFacto().to("cuda")

    # Freeze DiffFacto
    # occFacto.diffFacto.requires_grad = False

    # Get train data
    train_dataset, train_sampler = build_from_cfg(cfg.dataset.train, DATASETS, distributed=False)

    # Get Test data
    test_dataset, test_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

    
    log = open("dummy_train_log.txt", "w")



    # Conduct Train-Validation Loop

    # Conduct Final Test Loop



    # train_loss = train_loop(train_dataset, train_datasetNeg, optimizer, log)

    
import torch
import torch.optim as optim

# Assuming model is your neural network model
model = ...  # replace with your model
params = model.parameters()  # model parameters to optimize

# Initialize Adam optimizer
optimizer = optim.Adam(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

# Set up exponential decay scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Number of warm-up iterations and total epochs
num_warmup_iterations = 2000
num_epochs = ...  # total number of epochs for training

for epoch in range(num_epochs):
    for iteration, data in enumerate(training_data_loader):  # replace with your data loader
        # Warm-up phase
        if epoch * len(training_data_loader) + iteration < num_warmup_iterations:
            lr = 1e-4 * (epoch * len(training_data_loader) + iteration) / num_warmup_iterations
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training step
        inputs, labels = data  # replace with your data processing
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # replace with your loss function
        loss.backward()
        optimizer.step()
        
    # Apply exponential learning rate decay after each epoch
    if epoch % 500 == 499:  # Apply decay every 500 epochs
        scheduler.step()
'''