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
diffFacto = diffFacto.encoder.to("cuda")
diffFacto.eval()

# Create parameters for trainer class
train_folder = "occFacto2024RepResultsUppedPointsBatched"
eval_mesh_every = 20
bs_meshes = 2
visualize_every = 50
save_checkpoint = 25

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# Initialize the the Trainer class
trainer = Trainer(diffFacto, train_folder, eval_mesh_every, bs_meshes, visualize_every)

# Intialize the Model we want to Train
occFacto = OccFacto()
occFacto.to("cuda")

# Get train data
train_dataset, train_sampler = build_from_cfg(cfg.dataset.train, DATASETS, distributed=False)

# Get train data
validation_dataset, validation_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

# Get Loss Function
loss_f = nn.BCEWithLogitsLoss()

# Epochs
epochs = 3000

# Learning Rate Settings: Warm-up and Decay
learning_rate = 5e-4
warm_up_iter = 1000
lr_start = learning_rate / warm_up_iter

# Optimizer
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
# lambda_value = 0.0001 # L2 Regularization
# coptimizer = optim.Adam(occFacto.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=lambda_value)
optimizer = optim.Adam(occFacto.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)


# Warm Up
warmup = lr_scheduler.LinearLR(optimizer, 1 / warm_up_iter, 1, warm_up_iter)

# Decay
decay_factor = 0.9
decay_interval = 120
decay = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

# Logger File
log_file = os.path.join(train_folder, "training_stats.txt")

def log_out_file(epoch, train_metrics, validation_metrics, log, b_loss, b_iou, b_chamfer, best):
    # Train Info
    print(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Binary Accuracy: {train_metrics['Avg_Bin_Accuracy']} {best}")
    log.write(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Binary Accuracy: {train_metrics['Avg_Bin_Accuracy']} {best} \n")

    # Validation Info
    print(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Binary Accuracy: {validation_metrics['Avg_Bin_Accuracy']}, IOU: {validation_metrics['Avg_IOU']}, ChamferL1: {validation_metrics['Avg_ChamferL1']} {best}")
    log.write(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Binary Accuracy: {validation_metrics['Avg_Bin_Accuracy']}, IOU: {validation_metrics['Avg_IOU']}, ChamferL1: {validation_metrics['Avg_ChamferL1']} {best} \n")

    # Best info
    print(f"Best Loss: {b_loss}, Best IOU: {b_iou}, Best Chamfer: {b_chamfer}")

# Define the Train Loop
def train_loop(train_dataset, log_file, model, optimizer, scheduler, trainer, epochs):

    # Target Metrics
    best_loss, best_iou, best_chamfer = float("inf"), -1*float("inf"), float("inf")
    
    # Unpack
    warmup, decay = scheduler

    with open(log_file, 'w') as log:
        for epoch in range(epochs):
            
            # Initialize model train
            model.train()

            # Trackers
            loss_track = []
            in_accuracy_track = []
            iteration = 0
            bin_acc = 0
            accum = 0

            # Zero Grad
            optimizer.zero_grad()

            for pcds in tqdm(train_dataset):
                # Update iteration
                iteration += 1

                # Pass through diffFacto encoder to extract latents
                latents = diffFacto(pcds, device="cuda")
                latents = torch.cat(tuple(latents), dim=1)

                # Model outputs
                occPoints = pcds["occs"][0].to("cuda")
                occPreds = model(latents, occPoints) + 1e-10

                # Get Truths
                occTruths = pcds["occs"][1].to("cuda")

                # Loss:
                loss = loss_f(occPreds, occTruths) / 16
                loss.backward()
                accum += loss.item()

                # Get Metrics
                bin_acc += trainer.get_accuracy(occTruths, occPreds)
                # bin_acc = trainer.get_accuracy(occTruths, occPreds)
                
                if (iteration % 16 == 0) or (iteration == len(train_dataset)):
                    # Optimizer Step 
                    optimizer.step()
                    
                    # Warm Up Phase Per Spaghetti
                    # Built in Pytorch should stop when necessary
                    warmup.step() 

                    # Zero Grad
                    optimizer.zero_grad()

                    # Load Train Trackers
                    loss_track.append(accum / 16) # loss.item()
                    in_accuracy_track.append(bin_acc / 16) # bin_acc / 16

                    # Reset
                    bin_acc = 0
                    accum = 0

            # Learning Rate Scheduler
            if (epoch + 1) % decay_interval == 0: 
                decay.step()

            # Train Metrics
            train_metrics = {
                "Avg_Loss": np.mean(loss_track),
                "Avg_Bin_Accuracy": np.mean(in_accuracy_track)
            }

            # Get Validations Results / Metrics
            # Implicitly calls visualize functiom
            validation_metrics = trainer.validation(validation_dataset, model, diffFacto, loss_f, epoch)

            # Save Checkpoint
            if (epoch + 1) % save_checkpoint == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'validation_metrics': validation_metrics,
                }, f"{train_folder}/occFacto_epoch_{epoch}.pth")

            # Update Best Model
            if (validation_metrics["Avg_Loss"] < best_loss or
                validation_metrics["Avg_IOU"] > best_iou or
                validation_metrics["Avg_ChamferL1"] < best_chamfer):

                if validation_metrics["Avg_Loss"] < best_loss:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': validation_metrics
                    }, f"{train_folder}/occFacto_best_model_loss.pth")
                    best_loss = validation_metrics["Avg_Loss"]

                if validation_metrics["Avg_IOU"] > best_iou:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': validation_metrics
                    }, f"{train_folder}/occFacto_best_model_iou.pth")
                    best_iou = validation_metrics["Avg_IOU"]

                if validation_metrics["Avg_ChamferL1"] < best_chamfer:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': validation_metrics
                    }, f"{train_folder}/occFacto_best_model_chamfer.pth")
                    best_chamfer = validation_metrics["Avg_ChamferL1"]
                
                # Log Values and Print Values
                log_out_file(epoch, train_metrics, validation_metrics, log, best_loss, best_iou, best_chamfer, best="Best")

            else:
                # Log Values and Print Values
                log_out_file(epoch, train_metrics, validation_metrics, log, best_loss, best_iou, best_chamfer, best='')

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': validation_metrics
                }, f"{train_folder}/occFacto_final{epoch}.pth")

# Train the Model
train_loop(train_dataset, log_file, occFacto, optimizer, (warmup, decay), trainer, epochs)

