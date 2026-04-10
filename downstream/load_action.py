#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torchvision
import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, valid_loss):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def build_model(nclasses: int = 10, pretrained_weights: str = None):
    net = torchvision.models.convnext_large(weights='DEFAULT')
    input_emdim = net.classifier[2].in_features
    net.classifier[2] = nn.Linear(input_emdim, nclasses)
    
    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
        net.load_state_dict(state_dict, strict=False)
        
    net.cuda()
    return net

def setup_tensorboard(log_dir) -> torch.utils.tensorboard.SummaryWriter:
    return torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)

def calculate_metrics(probabilities, targets):
    predictions = (probabilities > 0.5).float()
    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()
    
    mAP = average_precision_score(targets_np, probabilities_np, average='macro')
    f1 = f1_score(targets_np, predictions_np, average='macro', zero_division=0)
    accuracy = accuracy_score(targets_np, predictions_np)
    recall = recall_score(targets_np, predictions_np, average='macro', zero_division=0)
    
    return mAP, f1, accuracy, recall

def valid(net: torch.nn.Module, valid_dl, loss_func, device='cuda'):
    valid_loss = 0
    all_probabilities = []
    all_targets = []

    net.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
        
        # Unpack the specific CholecT50 tuple structure to extract 'verbs' (index 2)
        for batch_idx, (inputs, (_, _, targets, _, _)) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = loss_func(outputs, targets)
            valid_loss += loss.item()
            display_loss = valid_loss / (batch_idx + 1)

            probabilities = torch.sigmoid(outputs)
            all_probabilities.append(probabilities)
            all_targets.append(targets)
            
            probabilities_current = torch.cat(all_probabilities)
            targets_current = torch.cat(all_targets)
            
            mAP, f1, accuracy, recall = calculate_metrics(probabilities_current, targets_current)

            pbar.set_description("Validation loss: %.3f | mAP: %.3f%%" % (display_loss, 100. * mAP))
    
    return display_loss, 100. * mAP, 100. * accuracy, 100. * f1, 100. * recall

if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module is not supposed to be run as an executable.')