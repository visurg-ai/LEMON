#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
from datetime import datetime
import torch.utils.data.sampler
import joblib
import lmdb
import torch.utils.tensorboard
import torchvision
import timm
import tqdm
import os
import h5py
import time
import random
import json
from sklearn.metrics import f1_score, accuracy_score


from .load_lmdb_autolaparo import Dataset, StringToIndexTransform, SubsetDataset, BalancedBatchSampler, EarlyStopping, build_model, valid, setup_tensorboard, build_preprocessing_transforms



def parse_cmdline_params():
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('--lr', required=True, type=float)  
    args.add_argument('--opt', required=True, type=str)
    args.add_argument('--nepochs', required=True, type=int)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--cpdir', required=True, type=str)
    args.add_argument('--logdir', required=True, type=str)
    args.add_argument('--cpint', required=True, type=int, default=10)
    args.add_argument('--kfold', required=True, type=int, default=1)
    args.add_argument('--lmdb', required=True, type=str)
    args.add_argument('--labels', required=True, type=str)
    args.add_argument('--resume', required=False, type=str, default=None)
    args.add_argument('--seed', required=False, type=int, default=None)
    args.add_argument('--pretrained', required=False, type=bool, default=None)
    args.add_argument('--pretrained-weights', required=False, type=str, default=None)
    
    return  args.parse_args()


def load_dataset(lmdb_path, label_path, output_json, class_mapping, train_preproc_tf = None, valid_preproc_tf = None, k: int = 5, train_bs: int = 512, valid_bs: int = 100, num_workers: int = 20, valid_size: float = 0.1):

    print('loading dataset')
    train_ds = Dataset(lmdb_path=lmdb_path, label_path=label_path)

    print("after loading dataset time:", datetime.now())
    num_train = len(train_ds)
    num_classes = len(class_mapping)
    indx=train_ds.index_img()

    class_ids = {f'{class_idx:02d}': [] for class_idx in range(1,22)}

    for idx in range(num_train):
        idx_name = indx[idx][0]
        class_ids[idx_name.split('_',1)[0]].append(idx)

    for class_idx in class_ids:
        np.random.shuffle(class_ids[class_idx])
    print("after shuffling class-index:", datetime.now())


    train_lists = []
    valid_lists = []
    test_lists = []

    rate = 0.5
    sepa = int(1 / rate)

    video_ids = list(class_ids.keys())

    for i in range(0, k):
        start = int(80/ sepa * i)
        gap_eachclass = int(80 / sepa)
        
        valid_fold = []
        test_fold = []
        
        for class_idx in video_ids[10: 14]:
            valid_fold += class_ids[class_idx]
        for class_idx in video_ids[14:21]:
            test_fold += class_ids[class_idx]
        
        valid_lists.append(valid_fold)
        test_lists.append(test_fold)
        
        np.random.shuffle(video_ids)

    total_list = list(range(len(train_ds)))
    for i in range(k):
        trainlist = list(set(total_list) - set(valid_lists[i]) - set(test_lists[i]))
        np.random.shuffle(trainlist)
        train_lists.append(trainlist)
        

    
    test_json={}
    for i in range(len(test_lists)):
        test_json[i] = {test:indx[test] for test in test_lists[i]}
    with open(f'{output_json}/test_fold.json','w') as file:
        json.dump(test_json, file, indent=4)


    return train_ds, train_lists, valid_lists


def build_optimizer(net, lr, opt: str = "adamw"):
    if opt.lower() == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    return optimizer


def resume(checkpoint_path, net, optimizer, scheduler, scaler):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('[ERROR] You want to resume from the last checkpoint, ' \
            + 'but there is not directory called "checkpoint"')
    
    state = torch.load(checkpoint_path)
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])

    # Update scheduler with saved params
    scheduler.load_state_dict(state['scheduler'])

    # Update scaler with saved params
    scaler.load_state_dict(state['scaler'])

    return state['lowest_valid_loss'], state['epoch'] + 1


def train(net: torch.nn, train_dl, loss_func, optimizer, scheduler, scaler, device='cuda'):

    net.train()
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))

    # Run forward-backward over all the samples
    train_loss = 0
    correct = 0
    total = 0
    targets_f1 = np.empty((0,))
    predicted_f1 = np.empty((0,))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Train with amp
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        targets_array = targets.detach().cpu().numpy()
        predicted_array = predicted.detach().cpu().numpy()
        targets_f1 = np.concatenate((targets_f1, targets_array))
        predicted_f1 = np.concatenate((predicted_f1, predicted_array))
        
        # Display loss and F1 score on the progress bar
        display_loss = train_loss / (batch_idx + 1)
        accuracy = accuracy_score(targets_f1,predicted_f1)
        F1_scores = f1_score(targets_f1,predicted_f1, average='macro')
        display_accuracy = 100. * accuracy
        display_F1 = 100. * F1_scores
        pbar.set_description("Training loss: %.3f | F1 score: %.3f%% | Accuracy score: %.3f%% (%d/%d) | LR: %.2E" % (display_loss, 
            display_F1, display_accuracy, correct, total, scheduler.get_last_lr()[0]))

    scheduler.step()

    return display_loss, display_F1


def main(args, train_dl=None, valid_dl=None):
    class_mapping = {'Other': 0, 'Picking-up the needle':1, 'Positioning the needle tip':2, 'Pushing the needle through the tissue':3, 'Pulling the needle out of the tissue':4, 'Tying a knot':5, 'Cutting the suture':6, 'Returning/dropping the needle':7}

    num_classes = len(class_mapping)
    
    if not os.path.isdir(args.cpdir):
        os.mkdir(args.cpdir)
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Prepare preprocessing layers
    train_preproc_tf, valid_preproc_tf = build_preprocessing_transforms()
    print("before loading dataset time:", datetime.now())
    
    # Get dataloaders for training and testing
    if train_dl is None and valid_dl is None:
        train_ds, train_lists, valid_lists = load_dataset(lmdb_path=args.lmdb, label_path=args.labels, output_json = args.cpdir, class_mapping = class_mapping, train_preproc_tf=train_preproc_tf, valid_preproc_tf=valid_preproc_tf, k=args.kfold, train_bs=args.bs, valid_bs=args.bs)

    # Create lists to store the losses and metrics
    train_loss_over_epochs = []
    train_F1_over_epochs = []
    valid_loss_over_epochs = []
    valid_F1_over_epochs = []
    
    # Training loop 
    for fold in range(args.kfold):
        # Build model
        net = build_model(nclasses=num_classes, pretrained_weights = args.pretrained_weights)

        # Enable multi-GPU support
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

        # Use cross-entropy loss
        loss_func = torch.nn.CrossEntropyLoss()
    
        # Build optimizer
        optimizer = build_optimizer(net, args.lr, args.opt)

        # Build LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)

        # Setup gradient scaler
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Setup Tensorboard
        writer = setup_tensorboard(args.logdir)
        
        # Resume from the last checkpoint if requested
        lowest_valid_loss = np.inf
        start_epoch = 0
        model_best = False
        if args.resume is not None:
            lowest_valid_loss, start_epoch = resume(args.resume, net, optimizer, scheduler, scaler)
            print('[INFO] Resuming from checkpoint:', start_epoch)
        
        train_loss_over_foldepochs = []
        train_F1_over_foldepochs = []
        valid_loss_over_foldepochs = []
        valid_F1_over_foldepochs = []    
        
        print("before loading subdataset time:", datetime.now())
        train_dataset = SubsetDataset(dataset=train_ds, indices=train_lists[fold], transform=train_preproc_tf)
        valid_dataset = SubsetDataset(dataset=train_ds, indices=valid_lists[fold], transform=valid_preproc_tf)

        # Create dataloaders
        print("before loading dataloader time:", datetime.now())
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=BalancedBatchSampler(data_indices=train_dataset.get_indices(), index_label=train_ds.index_img(), class_mapping=class_mapping), num_workers=8)
        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, num_workers=8)
        
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        for epoch in range(start_epoch, args.nepochs):  
            print(f"\n[INFO] Fold: {fold} Epoch: {epoch}")
            start = time.time()

            # Run a training epoch
            train_loss, train_F1 = train(net, train_dl, loss_func, optimizer, scheduler, scaler)

            # Run testing
            valid_loss, valid_F1, _, _, _ = valid(net, valid_dl, loss_func)

            # Update lowest validation loss
            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                model_best = True
            else:
                model_best = False

            # Update state
            state = {
                "net":               net.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "scheduler":         scheduler.state_dict(),
                "scaler":            scaler.state_dict(),
                "lowest_valid_loss": lowest_valid_loss,
                "epoch":             epoch,
            }

            # Save temporary checkpoint
            if epoch % args.cpint == 0 and epoch != 0:
                print('[INFO] Saving model for this epoch ...')
                checkpoint_path = os.path.join(args.cpdir, f"fold_{fold}_epoch_{epoch}.pt")
                torch.save(state, checkpoint_path) 
                print('[INFO] Saved.')

            # If it is the best model, let's save it too
            if model_best:
                print('[INFO] Saving best model ...')
                model_best_path = os.path.join(args.cpdir, f'fold_{fold}_model_best.pt')
                torch.save(state, model_best_path) 
                print('[INFO] Saved.')
        
            # Store training losses and metrics
            train_loss_over_foldepochs.append(train_loss)
            train_F1_over_foldepochs.append(train_F1)
        
            # Store validation losses and metrics
            valid_loss_over_foldepochs.append(valid_loss)
            valid_F1_over_foldepochs.append(valid_F1)

            # Log training losses and metrics in Tensorboard
            writer.add_scalar(f'Loss/{fold}/train', train_loss, epoch)
            writer.add_scalar(f'F1 score/{fold}/train', train_F1, epoch)
        
            # Log validation losses and metrics in Tensorboard
            writer.add_scalar(f'Loss/{fold}/valid', valid_loss, epoch)
            writer.add_scalar(f'F1 score/{fold}/valid', valid_F1, epoch)
       
            print(f"[INFO] Fold: {fold} Epoch: {epoch} finished.")
            
            # Check for early stopping
            if early_stopping(valid_loss):
                print(f'[INFO] Early stopping triggered at epoch {epoch} for fold {fold}.')
                break

        
        #Store every fold training losses and metrics    
        train_loss_over_epochs.append(train_loss_over_foldepochs)
        train_F1_over_epochs.append(train_F1_over_foldepochs)
        
        #Store every fold validation losses and metrics
        valid_loss_over_epochs.append(valid_loss_over_foldepochs)
        valid_F1_over_epochs.append(valid_F1_over_foldepochs)
    

if __name__ == '__main__':
    # Parse command line parameters
    args = parse_cmdline_params()

    # Call main training function
    main(args)
