#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import timm
import cv2
import h5py
from copy import deepcopy
import tqdm
import os
import json
import pickle
import lmdb
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor, Lambda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import stdev
from .load_lmdb_autolaparo import StringToIndexTransform, build_preprocessing_transforms, build_model

class TestDataset(VisionDataset):
    def __init__(
        self,
        lmdb_path: str,
        labels: str,
        fold: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.labels = labels
        self.fold = str(fold)
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        
        with open(self.labels, 'r') as file:
            label = json.load(file)
            label = label[self.fold]
            num_list = list(label.keys())
            self.length = len(num_list)
            
        with self.env.begin() as txn:
            for num in num_list:
                name = label[num][0]
                target = label[num][1]
                
                img_key = f'img_{name}'.encode('utf-8')
                img_bytes = txn.get(img_key)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)
                
                self.data.append(img)
                self.targets.append(target)

    
    def __getitem__(self, index: int):
      
        image = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return self.length



class StringToIndexTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, label):
        return self.class_mapping[label]



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
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            #print(outputs.shape)
            
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
            different_indices = np.where(targets_f1 != predicted_f1)[0]
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




def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('--lmdb', required=True, type=str)
    args.add_argument('--labels', required=True, type=str)
    args.add_argument('--models', required=False, type=str)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--kfold', required=True, type=int)
    
    return  args.parse_args()


def load_dataset(lmdb_path, labels, fold, test_preproc_tf, bs: int, num_workers: int = 2):
    class_mapping = {'Other': 0, 'Picking-up the needle':1, 'Positioning the needle tip':2, 'Pushing the needle through the tissue':3, 'Pulling the needle out of the tissue':4, 'Tying a knot':5, 'Cutting the suture':6, 'Returning/dropping the needle':7}
   
    num_classes = len(class_mapping)

    test_ds = TestDataset(lmdb_path=lmdb_path, labels=labels, fold=fold, transform=test_preproc_tf)
    
    # Create dataloader
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, 
                                          shuffle=False, 
                                          num_workers=num_workers)

    return test_dl, num_classes


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    _, test_preproc_tf = build_preprocessing_transforms()

    test_F1score = []
    test_accuracyscore = []
    test_prescore = []
    test_recallscore = []
    for fold_number in range(args.kfold):
        
        # Get dataloaders for training and testing
        test_dl, num_classes = load_dataset(lmdb_path = args.lmdb, labels = args.labels, fold = fold_number, test_preproc_tf = test_preproc_tf, bs = args.bs)

        # Build model
        model = build_model(nclasses = num_classes)

        # Enable multi-GPU support
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

        # Load weights from file
        cp = os.path.join(args.models, f'fold_{fold_number}_model_best.pt')
        state = torch.load(cp, map_location=torch.device('cuda'))
        net = state['net']
        model.load_state_dict(net)

        # Use cross-entropy loss
        loss_func = torch.nn.CrossEntropyLoss()
    
        # Run testing
        avg_loss, test_F1, test_accuracy, test_precision, test_recall = valid(model, test_dl, loss_func)
        test_F1score.append(float(test_F1))
        test_accuracyscore.append(float(test_accuracy))
        test_prescore.append(float(test_precision))
        test_recallscore.append(float(test_recall))
    
        # Print results report
        print(f'[INFO] fold {fold_number} loss: {avg_loss}')
        print(f'[INFO] fold {fold_number} F1 score: {test_F1:.3f}%, Accuracy: {test_accuracy:.3f}%, Precision:{test_precision:.3f}%, Recall: {test_recall:.3f}% ')
       
    # Calculate the average
    print(f'[INFO] Average F1 score: {(sum(test_F1score) / len(test_F1score)):.2f}%, stdev: {stdev(test_F1score)} ')
    print(f'[INFO] Average accuracy score: {(sum(test_accuracyscore) / len(test_accuracyscore)):.2f}%, stdev: {stdev(test_accuracyscore)} ')
    print(f'[INFO] Average precision score: {(sum(test_prescore) / len(test_prescore)):.2f}%, stdev: {stdev(test_prescore)} ')
    print(f'[INFO] Average recall score: {(sum(test_recallscore) / len(test_recallscore)):.2f}%, stdev: {stdev(test_recallscore)} ')
    


if __name__ == '__main__':
    main()
