#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import cv2
import lmdb
import tqdm
import torch
import torch.nn as nn
import torchvision
from typing import Callable, Optional
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

class Dataset(VisionDataset):
    def __init__(
        self,
        lmdb_path: str,
        label_path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=lmdb_path, transform=transform)
        self.lmdb = lmdb_path
        self.label_path = label_path
        self.index_imgjson = {}
        self.data = []
        self.targets = []
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        self.idx = 0
        
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
            image_names = list(labels.keys())
            
        for name in tqdm.tqdm(image_names, desc="Loading Dataset"):
            img_key = f'img_{name}'.encode('utf-8')
            self.data.append(img_key)
            label = labels[name]
            self.targets.append(label)
            self.index_imgjson[self.idx] = (name, label)
            self.idx += 1

    def __getitem__(self, index: int):
        with self.env.begin() as txn:
            img_bytes = txn.get(self.data[index])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return img, target

    def __len__(self) -> int:
        return self.idx

    def index_img(self):
        return self.index_imgjson

class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.dataset[real_idx]
        
        # Avoid double transformation if the parent dataset already applied it, 
        # but in this architecture, we apply it here.
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def get_indices(self):
        return self.indices

class TestDataset(VisionDataset):
    def __init__(
        self,
        lmdb_path: str,
        labels: str,
        fold: int,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=lmdb_path, transform=transform)
        self.lmdb = lmdb_path
        self.labels = labels
        self.fold = str(fold)
        self.data = []
        self.targets = []
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        
        with open(self.labels, 'r') as file:
            label_data = json.load(file)
            fold_labels = label_data[self.fold]
            num_list = list(fold_labels.keys())
            self.length = len(num_list)
            
        with self.env.begin() as txn:
            for num in tqdm.tqdm(num_list, desc=f"Loading Test Fold {fold}"):
                name = fold_labels[num][0]
                target = fold_labels[num][1]
                
                img_key = f'img_{name}'.encode('utf-8')
                img_bytes = txn.get(img_key)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)
                
                self.data.append(img)
                self.targets.append(target)
    
    def __getitem__(self, index: int):
        image = self.data[index]
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return self.length

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

def build_model(nclasses: int = 7, pretrained_weights: str = None):
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

def build_preprocessing_transforms():
    train_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    valid_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return train_preproc_tf, valid_preproc_tf

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
        for batch_idx, (inputs, targets) in pbar:
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