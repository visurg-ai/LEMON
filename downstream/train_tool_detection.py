#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import json
import random
import numpy as np
import torch
import tqdm
from datetime import datetime

from .load_lmdb_multilabel import (
    Dataset, SubsetDataset, EarlyStopping, calculate_metrics, valid, 
    build_model, build_preprocessing_transforms, setup_tensorboard
)

def parse_cmdline_params():
    args = argparse.ArgumentParser(description='PyTorch Tool Detection Training.')
    args.add_argument('--lr', required=True, type=float)  
    args.add_argument('--opt', required=True, type=str)
    args.add_argument('--nepochs', required=True, type=int)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--cpdir', required=True, type=str)
    args.add_argument('--logdir', required=True, type=str)
    args.add_argument('--cpint', required=True, type=int)
    args.add_argument('--kfold', required=True, type=int)
    args.add_argument('--lmdb', required=True, type=str)
    args.add_argument('--labels', required=True, type=str)
    args.add_argument('--resume', required=False, type=str, default=None)
    args.add_argument('--seed', required=False, type=int, default=None)
    args.add_argument('--pretrained-weights', required=False, type=str, default=None)
    return args.parse_args()

def split_dataset(train_ds, k, output_json):
    num_train = len(train_ds)
    indx = train_ds.index_img()

    video_ids_dict = {}
    for idx in range(num_train):
        idx_name = indx[idx][0]
        video_id = idx_name.split('_')[0]
        if video_id not in video_ids_dict:
            video_ids_dict[video_id] = []
        video_ids_dict[video_id].append(idx)

    for vid in video_ids_dict:
        np.random.shuffle(video_ids_dict[vid])

    video_ids = list(video_ids_dict.keys())
    np.random.shuffle(video_ids)

    train_lists, valid_lists, test_lists = [], [], []
    fold_size = len(video_ids) // k

    test_json = {}
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i != k - 1 else len(video_ids)
        
        valid_vid_subset = video_ids[start_idx:end_idx]
        test_vid_subset = video_ids[(start_idx + fold_size) % len(video_ids): (start_idx + 2 * fold_size) % len(video_ids)]
        
        valid_fold, test_fold, train_fold = [], [], []
        
        for vid in valid_vid_subset:
            valid_fold.extend(video_ids_dict[vid])
        for vid in test_vid_subset:
            test_fold.extend(video_ids_dict[vid])
            
        train_vids = list(set(video_ids) - set(valid_vid_subset) - set(test_vid_subset))
        for vid in train_vids:
            train_fold.extend(video_ids_dict[vid])

        np.random.shuffle(train_fold)
        train_lists.append(train_fold)
        valid_lists.append(valid_fold)
        test_lists.append(test_fold)
        
        test_json[i] = {test_idx: indx[test_idx] for test_idx in test_fold}

    with open(os.path.join(output_json, 'test_fold.json'), 'w') as file:
        json.dump(test_json, file, indent=4)

    return train_lists, valid_lists

def build_optimizer(net, lr, opt: str = "adam"):
    if opt.lower() == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    elif opt.lower() == "sgd":
        return torch.optim.SGD(net.parameters(), lr=lr)

def resume(checkpoint_path, net, optimizer, scheduler, scaler):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('[ERROR] Checkpoint not found.')
    state = torch.load(checkpoint_path)
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    return state['lowest_valid_loss'], state['epoch'] + 1

def train(net, train_dl, loss_func, optimizer, scheduler, scaler, device='cuda'):
    net.train()
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    
    train_loss = 0
    all_probabilities = []
    all_targets = []
    
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        display_loss = train_loss / (batch_idx + 1)

        probabilities = torch.sigmoid(outputs)
        all_probabilities.append(probabilities)
        all_targets.append(targets)
        
        probabilities_current = torch.cat(all_probabilities)
        targets_current = torch.cat(all_targets)
        mAP, _, _, _ = calculate_metrics(probabilities_current, targets_current)

        pbar.set_description("Training loss: %.3f | mAP: %.3f%% | LR: %.2E" % (
            display_loss, 100. * mAP, scheduler.get_last_lr()[0]
        ))

    scheduler.step()
    return display_loss, 100. * mAP

def main():
    args = parse_cmdline_params()
    class_mapping = {'Grasper': 0, 'Bipolar': 1, 'Hook': 2, 'Scissors': 3, 'Clipper': 4, 'Irrigator': 5, 'SpecimenBag': 6}					
    num_classes = len(class_mapping)

    if not os.path.isdir(args.cpdir):
        os.makedirs(args.cpdir)

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train_preproc_tf, valid_preproc_tf = build_preprocessing_transforms()
    train_ds = Dataset(lmdb_path=args.lmdb, label_path=args.labels, transform=None)
    train_lists, valid_lists = split_dataset(train_ds, k=args.kfold, output_json=args.cpdir)

    for fold in range(args.kfold):
        net = build_model(nclasses=num_classes, pretrained_weights=args.pretrained_weights)
        net = torch.nn.DataParallel(net)
        loss_func = torch.nn.BCEWithLogitsLoss()
        optimizer = build_optimizer(net, args.lr, args.opt)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        writer = setup_tensorboard(args.logdir)

        lowest_valid_loss = np.inf
        start_epoch = 0
        if args.resume is not None:
            lowest_valid_loss, start_epoch = resume(args.resume, net, optimizer, scheduler, scaler)

        train_dataset = SubsetDataset(dataset=train_ds, indices=train_lists[fold], transform=train_preproc_tf)
        valid_dataset = SubsetDataset(dataset=train_ds, indices=valid_lists[fold], transform=valid_preproc_tf)

        train_dl = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.bs, num_workers=10)
        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, num_workers=10)
        
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(start_epoch, args.nepochs):  
            print(f"\n[INFO] Fold: {fold} Epoch: {epoch}")
            train_loss, train_map = train(net, train_dl, loss_func, optimizer, scheduler, scaler)
            valid_loss, valid_map, _, _, _ = valid(net, valid_dl, loss_func)

            model_best = False
            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                model_best = True

            state = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "lowest_valid_loss": lowest_valid_loss,
                "epoch": epoch,
            }

            if epoch % args.cpint == 0 and epoch != 0:
                torch.save(state, os.path.join(args.cpdir, f"fold_{fold}_epoch_{epoch}.pt")) 

            if model_best:
                torch.save(state, os.path.join(args.cpdir, f'fold_{fold}_model_best.pt')) 

            writer.add_scalar(f'Loss/{fold}/train', train_loss, epoch)
            writer.add_scalar(f'mAP/{fold}/train', train_map, epoch)
            writer.add_scalar(f'Loss/{fold}/valid', valid_loss, epoch)
            writer.add_scalar(f'mAP/{fold}/valid', valid_map, epoch)
            
            if early_stopping(valid_loss):
                print(f'[INFO] Early stopping triggered at epoch {epoch} for fold {fold}.')
                break

if __name__ == '__main__':
    main()