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
trainer = Trainer()

# Intialize the Model we want to Train
occFacto = OccFacto().to("cuda")

# Get train data
train_dataset, train_sampler = build_from_cfg(cfg.dataset.train, DATASETS, distributed=False)

# Get train data
validation_dataset, validation_sampler = build_from_cfg(cfg.dataset.val, DATASETS, distributed=False)

# Get test data
# test_dataset, test_sampler = build_from_cfg(cfg.dataset.test, DATASETS, distributed=False)

# Loss and Training Parameters: Based on Spaghetti Paper 
# Loss Function
# p_w = torch.tensor(15.1908)
# loss_f = nn.BCEWithLogitsLoss(pos_weight = p_w)
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
optimizer = optim.Adam(occFacto.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

# Warm Up
warmup = lr_scheduler.LinearLR(optimizer, 1 / warm_up_iter, 1, warm_up_iter)

# Decay
decay_factor = 0.9
decay_interval = 800
decay = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

# Logger File
log_file = "occFactoDiffFreezeTraining3/training_stats.txt"

# Define the Train Loop
def train_loop(train_dataset, log_file, model, optimizer, scheduler, trainer, epochs):
    best_loss = float("inf")
    
    warmup, scheduler = scheduler

    with open(log_file, 'w') as log:
        for epoch in range(epochs):

            # Initialize model train
            model.train()

            # Trackers
            loss_track = []
            accuracy_track = []
            iou_track = []
            chamfer_track = []
            normal_track = []

            iteration = 0
            for pcds in tqdm(train_dataset):
                # Update iteration
                iteration += 1
                
                # Zero Grad
                optimizer.zero_grad()

                # Pass through diffFacto encoder to extract latents
                latents = diffFacto(pcds, device="cuda")
                latents = torch.cat(tuple(latents), dim=1)

                # Model outputs
                occPoints = pcds["occs"][0].to("cuda")
                occPreds = model(latents, occPoints)

                # Get Truths
                occTruths = pcds["occs"][1].to("cuda")

                # Loss:
                loss = loss_f(occPreds, occTruths)
                loss.backward()

                # Optimizer Step
                optimizer.step()

                # Warm Up Phase Per Spaghetti
                if epoch * len(train_dataset) + iteration < warm_up_iter:
                    warmup.step() 

                # Learning Rate Scheduler
                if (epoch + 1) % decay_interval == 0: 
                    scheduler.step()

                # Get Metrics
                param_accuracy = (occTruths, occPreds)
                metrics = trainer.metrics(param_accuracy)

                # Load Trackers
                loss_track.append(loss.item())
                accuracy_track.append(metrics["accuracy"])
                # iou_track.append(metrics["iou"])
                # chamfer_track.append(metrics["chamfer"])
                # normal_track.append(metrics["normal"])

            # Train Metrics
            train_metrics = {
                "Avg_Loss": np.mean(loss_track),
                "Avg_Accuracy": np.mean(accuracy_track)
            }

            # Get Validations Results / Metrics
            validation_metrics = trainer.validation(validation_dataset, model, diffFacto, loss_f, epoch)

            # Save Checkpoint
            # trainer.backup()
            if (epoch + 1) % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'validation_metrics': validation_metrics,
                }, f"occFactoDiffFreezeTraining2/occFacto_epoch_{epoch}.pth")

            # Save visualization
            
            # Save a visualization of a random subset EVERY???

            # Update Best Model
            if validation_metrics["Avg_Loss"] < best_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': validation_metrics["Avg_Loss"]
                }, f"occFactoDiffFreezeTraining2/occFacto_best_model.pth")
                print(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Accuracy: {train_metrics['Avg_Accuracy']}, Best")
                log.write(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Accuracy: {train_metrics['Avg_Accuracy']}, Best \n")
                print(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Accuracy: {validation_metrics['Avg_Accuracy']}, Best")
                log.write(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Accuracy: {validation_metrics['Avg_Accuracy']}, Best \n")
                
            # trainer.update_best_model(train_metrics, validation_metrics, model)
            else:
                # Log Values and Print Values
                print(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Accuracy: {train_metrics['Avg_Accuracy']}")
                log.write(f"Train - Epoch: {epoch + 1}, Loss: {train_metrics['Avg_Loss']}, Accuracy: {train_metrics['Avg_Accuracy']} \n")
                print(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Accuracy: {validation_metrics['Avg_Accuracy']}")
                log.write(f"Validation - Epoch: {epoch + 1}, Loss: {validation_metrics['Avg_Loss']}, Accuracy: {validation_metrics['Avg_Accuracy']} \n")

        # Save params after training
        # trainer.save_checkpoint(final=True)
        torch.save(model.state_dict(), f"occFactoDiffFreezeTraining3/occFacto_final{epoch}.pth")

# Train the Model
train_loop(train_dataset, log_file, occFacto, optimizer, (warmup, decay), trainer, epochs)

