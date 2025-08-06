#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
import random
from copy import deepcopy
import torch.utils.tensorboard
import torchvision
import timm
import tqdm
import os
import lmdb
import json
import cv2
import h5py
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor, Lambda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class Dataset(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        label_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.label_path = label_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.index_imgjson = {}
        self.data = []
        self.targets = []
        self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        self.idx = 0
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
            image_names = list(labels.keys())
        with self.env.begin() as txn:
            for name in tqdm.tqdm(image_names, total = len(image_names)):
                img_key = f'img_{name}'.encode('utf-8')
                
                img_bytes = txn.get(img_key)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)

                self.data.append(img)
                label = int(labels[name])
                self.targets.append(label)
  
                self.index_imgjson[self.idx] = (name, label)
                self.idx += 1


    def _get_num_samples(self, txn):
        num_samples = 0
        cursor = txn.cursor()
        for key, _ in cursor:
            if key.startswith(b'label_'):
                num_samples += 1
        return num_samples


    def __getitem__(self, index: int):
        img = self.data[index]

        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return self.idx

    def index_img(self):
        return self.index_imgjson



class StringToIndexTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, label):
        return self.class_mapping[label]


def setup_tensorboard(log_dir) -> torch.utils.tensorboard.SummaryWriter:

    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
    return writer



class SubsetDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.index_imgjson = {}
        self.data = []
        self.original_idx = []
        self.targets = []
        for index in tqdm.tqdm(range(len(self.indices)), total = len(self.indices)):
            original_idx = self.indices[index]
            
            sample = dataset[original_idx]
            self.data.append(sample[0])
            self.targets.append(sample[1])


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        
        return img, target

    
    def get_indices(self):
        return self.indices
    

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_indices, index_label, class_mapping, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        self.class_mapping = class_mapping
        
        for idx in range(0, len(data_indices)):
            label = self._get_label(idx=idx, index_label=index_label, indices=data_indices)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, idx, index_label, indices, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            original_idx = indices[idx]
            class_name = index_label[original_idx][1]
            return class_name

    def __len__(self):
        return self.balanced_max*len(self.keys)


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




def build_model(nclasses: int = 2, pretrained: bool = True):

    net = torchvision.models.convnext_large(weights='DEFAULT')
    
    for param in net.features.parameters():
        param.requires_grad = True
    
    input_emdim = net.classifier[2].in_features
    net.classifier[2] = nn.Identity()
    
    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict = state_dict['teacher']


        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
        msg = net.load_state_dict(state_dict, strict=False)
        print(msg, input_emdim)
        
    net.classifier[2] = nn.Linear(input_emdim, nclasses)
    
    net.cuda()


    return net

def build_preprocessing_transforms(size: int = 384, randaug_n: int = 2, 
                                   randaug_m: int = 14):

    # Preprocessing for training
    train_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=10, fill=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # Preprocessing for testing
    valid_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return train_preproc_tf, valid_preproc_tf


def setup_tensorboard(log_dir) -> torch.utils.tensorboard.SummaryWriter:

    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
    return writer


def valid(net: torch.nn, valid_dl, loss_func, device='cuda'):

    valid_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    targets_f1 = np.empty((0,))
    predicted_f1 = np.empty((0,))

    net.eval()
    with torch.no_grad():
        # Create progress bar
        pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
        
        # Loop over testing data points
        for batch_idx, (inputs, targets) in pbar:
            # Perform inference
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            # Store top class prediction and ground truth label
            y_true.append(targets[0].item())
            y_pred.append(torch.argmax(outputs).item())

            # Compute losses and metrics
            loss = loss_func(outputs, targets)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)

            targets_array = targets.detach().cpu().numpy()
            predicted_array = predicted.detach().cpu().numpy()
            targets_f1 = np.concatenate((targets_f1, targets_array))
            predicted_f1 = np.concatenate((predicted_f1, predicted_array))
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Display loss and top-1 f1 score on the progress bar 
            display_loss = valid_loss / (batch_idx + 1)
            accuracy = accuracy_score(targets_f1,predicted_f1)
            precision = precision_score(targets_f1,predicted_f1, average='macro')
            recall = recall_score(targets_f1,predicted_f1, average='macro')
            F1_scores = f1_score(targets_f1,predicted_f1, average='macro')
            display_accuracy = 100. * accuracy
            display_precision = 100. * precision
            display_recall = 100. * recall
            display_F1 = 100. * F1_scores
            pbar.set_description("Validation loss: %.3f | F1 score: %.3f%% (%d/%d)" % (display_loss,
                display_F1, correct, total))
    
    return display_loss, display_F1, display_accuracy, display_precision, display_recall 


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module is not supposed to be run as an executable.')
