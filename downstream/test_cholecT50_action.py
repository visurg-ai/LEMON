#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from statistics import stdev

from load_action import valid, build_model
from .dataloader_cholecT50 import CholecT50

def parse_cmdline_params():
    args = argparse.ArgumentParser(description='PyTorch Action Recognition Testing.')
    args.add_argument('--dataset-dir', required=True, type=str)
    args.add_argument('--models', required=True, type=str)
    args.add_argument('--bs', required=True, type=int)
    args.add_argument('--kfold', required=True, type=int)
    args.add_argument('--pretrained-weights', required=False, type=str, default=None)
    return args.parse_args()

def load_dataset(dataset_dir, fold, bs: int, num_workers: int = 4):
    class_mapping = {"grasp": 0, "retract": 1, "dissect": 2, "coagulate": 3, "clip": 4, 
                     "cut": 5, "aspirate": 6, "irrigate": 7, "pack": 8, "null_verb": 9}
    num_classes = len(class_mapping)

    dataset = CholecT50( 
        test=True,
        dataset_dir=dataset_dir, 
        dataset_variant="cholect50-crossval",
        test_fold=fold + 1,
        augmentation_list=['original', 'vflip', 'hflip']
    )

    _, _, test_dataset = dataset.build()
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    return test_dl, num_classes

def main():
    args = parse_cmdline_params()

    test_F1score = []
    test_accuracyscore = []
    test_mapscore = []
    test_recallscore = []

    for fold_number in range(args.kfold):
        test_dl, num_classes = load_dataset(
            dataset_dir=args.dataset_dir, 
            fold=fold_number, 
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