#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from statistics import stdev

from .load_lmdb_multilabel import (
    TestDataset, valid, build_model, build_preprocessing_transforms
)

def parse_cmdline_params():
    args = argparse.ArgumentParser(description='PyTorch Tool Detection Testing.')
    args.add_argument('--lmdb', required=True, type=str)
    args.add_argument('--labels', required=True, type=str)
    args.add_argument('--models', required=True, type=str)
    args.add_argument('--pretrained-weights', default=None, required=False, type=str)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--kfold', required=True, type=int)
    return args.parse_args()

def load_dataset(lmdb_path, labels, fold, test_preproc_tf, bs: int, num_workers: int = 4):
    class_mapping = {'Grasper': 0, 'Bipolar': 1, 'Hook': 2, 'Scissors': 3, 'Clipper': 4, 'Irrigator': 5, 'SpecimenBag': 6}
    num_classes = len(class_mapping)
    test_ds = TestDataset(lmdb_path=lmdb_path, labels=labels, fold=fold, transform=test_preproc_tf)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    return test_dl, num_classes

def main():
    args = parse_cmdline_params()
    _, test_preproc_tf = build_preprocessing_transforms()

    test_F1score = []
    test_accuracyscore = []
    test_mapscore = []
    test_recallscore = []

    for fold_number in range(args.kfold):
        test_dl, num_classes = load_dataset(
            lmdb_path=args.lmdb, 
            labels=args.labels, 
            fold=fold_number, 
            test_preproc_tf=test_preproc_tf, 
            bs=args.bs
        )

        model = build_model(nclasses=num_classes, pretrained_weights=args.pretrained_weights)
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

        cp = os.path.join(args.models, f'fold_{fold_number}_model_best.pt')
        state = torch.load(cp, map_location=torch.device('cuda'))
        model.load_state_dict(state['net'])

        loss_func = torch.nn.BCEWithLogitsLoss()
    
        avg_loss, test_map, test_accuracy, test_F1, test_recall = valid(model, test_dl, loss_func)
        
        test_mapscore.append(float(test_map))
        test_accuracyscore.append(float(test_accuracy))
        test_F1score.append(float(test_F1))
        test_recallscore.append(float(test_recall))
    
        print(f'[INFO] fold {fold_number} loss: {avg_loss:.4f}')
        print(f'[INFO] fold {fold_number} mAP: {test_map:.3f}%, F1: {test_F1:.3f}%, Accuracy: {test_accuracy:.3f}%, Recall: {test_recall:.3f}%')

    print(f'[INFO] Average mAP: {(sum(test_mapscore) / len(test_mapscore)):.2f}%, stdev: {stdev(test_mapscore):.2f}')
    print(f'[INFO] Average F1 score: {(sum(test_F1score) / len(test_F1score)):.2f}%, stdev: {stdev(test_F1score):.2f}')
    print(f'[INFO] Average Accuracy: {(sum(test_accuracyscore) / len(test_accuracyscore)):.2f}%, stdev: {stdev(test_accuracyscore):.2f}')
    print(f'[INFO] Average Recall: {(sum(test_recallscore) / len(test_recallscore)):.2f}%, stdev: {stdev(test_recallscore):.2f}')

if __name__ == '__main__':
    main()