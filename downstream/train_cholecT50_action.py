#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import random
import numpy as np
import torch
import tqdm

from .load_action import (
    EarlyStopping, calculate_metrics, valid, 
    build_model, setup_tensorboard
)
from .dataloader_cholecT50 import CholecT50

def parse_cmdline_params():
    args = argparse.ArgumentParser(description='PyTorch Action Recognition Training.')
    args.add_argument('--lr', required=True, type=float)  
    args.add_argument('--opt', required=True, type=str)
    args.add_argument('--nepochs', required=True, type=int)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--cpdir', required=True, type=str)
    args.add_argument('--logdir', required=True, type=str)
    args.add_argument('--cpint', required=True, type=int)
    args.add_argument('--kfold', required=True, type=int)
    args.add_argument('--dataset-dir', required=True, type=str)
    args.add_argument('--resume', required=False, type=str, default=None)
    args.add_argument('--seed', required=False, type=int, default=None)
    args.add_argument('--pretrained-weights', required=False, type=str, default=None)
    return args.parse_args()

def build_optimizer(net, lr, opt: str = "adam"):
    if opt.lower() == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
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
    
    # Unpack the specific CholecT50 tuple structure to extract 'verbs' (index 2)
    for batch_idx, (inputs, (_, _, targets, _, _)) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        probabilities = torch.sigmoid(outputs)
        all_probabilities.append(probabilities)
        all_targets.append(targets)
        
        probabilities_current = torch.cat(all_probabilities)
        targets_current = torch.cat(all_targets)
        mAP, _, _, _ = calculate_metrics(probabilities_current, targets_current)
        
        train_loss += loss.item()
        display_loss = train_loss / (batch_idx + 1)
        
        pbar.set_description("Training loss: %.3f | mAP: %.3f%% | LR: %.2E" % (
            display_loss, 100. * mAP, scheduler.get_last_lr()[0]
        ))

    scheduler.step()
    return display_loss, 100. * mAP

def main():
    args = parse_cmdline_params()
    class_mapping = {"grasp": 0, "retract": 1, "dissect": 2, "coagulate": 3, "clip": 4, 
                     "cut": 5, "aspirate": 6, "irrigate": 7, "pack": 8, "null_verb": 9}
    num_classes = len(class_mapping)

    if not os.path.isdir(args.cpdir):
        os.makedirs(args.cpdir)
    
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    for fold in range(args.kfold):
        net = build_model(nclasses=num_classes, pretrained_weights=args.pretrained_weights)
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

        loss_func = torch.nn.BCEWithLogitsLoss()
        optimizer = build_optimizer(net, args.lr, args.opt)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        writer = setup_tensorboard(args.logdir)
        
        lowest_valid_loss = np.inf
        start_epoch = 0
        if args.resume is not None:
            lowest_valid_loss, start_epoch = resume(args.resume, net, optimizer, scheduler, scaler)

        dataset = CholecT50( 
            test=False,
            dataset_dir=args.dataset_dir, 
            dataset_variant="cholect50-crossval",
            test_fold=fold + 1,
            augmentation_list=['original', 'vflip', 'hflip']
        )

        train_dataset, val_dataset, _ = dataset.build()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
        
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(start_epoch, args.nepochs):  
            print(f"\n[INFO] Fold: {fold} Epoch: {epoch}")
            train_loss, train_map = train(net, train_dataloader, loss_func, optimizer, scheduler, scaler)
            valid_loss, valid_map, _, _, _ = valid(net, val_dataloader, loss_func)

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