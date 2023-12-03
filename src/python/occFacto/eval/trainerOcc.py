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

from torch.optim import lr_scheduler

# from occFacto.eval.generate import Generator


class Trainer():

    def __init__(self):
        pass

    def train_step(self, encoder, decoder, pcds):

        # Pass through diffFacto encoder to extract latents
        latents = encoder(pcds, device="cuda")
        latents = torch.cat(tuple(latents), dim=1)

        # Model outputs
        occPoints = pcds["occs"][0]
        occPreds = decoder(latents, occPoints)

        return occPreds

    def validation(self, data, model, diffFacto, loss_f):
        model.eval()

        # Trackers
        loss_track = []
        accuracy_track = []

        for ex in data:
            # Pass through diffFacto encoder to extract latents
            with torch.no_grad():
                latents = diffFacto(ex, device="cuda")
                latents = torch.cat(tuple(latents), dim=1)

                # Model outputs
                occPoints = ex["occs"][0].to("cuda")
                occPreds = model(latents, occPoints)

                # Get Truths
                occTruths = ex["occs"][1].to("cuda")

                # Loss
                loss_track.append(loss_f(occPreds, occTruths).item())

                # Accuracy
                accuracy_track.append(self.metrics((occTruths, occPreds))["accuracy"])

        # Val Metrics
        val_metrics = {
            "Avg_Loss": np.mean(loss_track),
            "Avg_Accuracy": np.mean(accuracy_track)
        }

        return val_metrics


    def log(self):
        pass

    def get_accuracy(self, params):
        truth, preds = params
        out = torch.sigmoid(preds)
        bin = (out >= 0.5).int()
        correct_predictions = torch.sum(bin == truth)
        total_predictions = bin.numel()
        accuracy = correct_predictions.item() / total_predictions
        return accuracy

    def metrics(self, accuracy_p=None, iou_p=None, chamfer_p=None, normal_p=None):
        metrics = {
            "accuracy": None,
            "iou": None,
            "chamfer": None,
            "normal": None
        }

        if accuracy_p:
            metrics["accuracy"] = self.get_accuracy(accuracy_p)
        if iou_p:
            metrics["iou"] = self.get_iou(iou_p)
        if chamfer_p:
            metrics["chamfer"] = self.get_iou(chamfer_p)
        if normal_p:
            metrics["normal"] = self.get_iou(normal_p)
        
        return metrics

    def save_visualization(self):
        pass

    def backup(self):
        pass

    def update_best_model(self):
        pass

    