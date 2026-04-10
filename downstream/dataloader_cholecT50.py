#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

class CholecT50:
    def __init__(self, 
                 test=False,
                 dataset_dir=None, 
                 dataset_variant="cholect50-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip'],
                 normalize=True):
        
        self.test = test
        self.normalize = normalize
        self.dataset_dir = dataset_dir
        
        video_split = self.split_selector(case=dataset_variant)
        if 'crossval' in dataset_variant:
            train_videos = sum([v for k, v in video_split.items() if k != test_fold], [])
            test_videos = sum([v for k, v in video_split.items() if k == test_fold], [])
            val_videos = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            train_videos = video_split['train']
            val_videos = video_split['val']
            test_videos = video_split['test']
        
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]

        self.augmentations = {
            'original': lambda x: x,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast_jitter': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'sharpness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'autocontrast': transforms.RandomAutocontrast(p=0.5),
        }
        
        self.augmentation_list = [self.augmentations[aug] for aug in augmentation_list]
        trainform, testform = self.transform()

        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        
        self.target_transform = self.to_binary
        
        self.build_test_dataset(testform)
        if not self.test:
            self.build_train_dataset(trainform)
            self.build_val_dataset(testform)

    def split_selector(self, case='cholect50-crossval'):
        switcher = {
            'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
        }
        return switcher.get(case)

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        op_test = [transforms.Resize((224, 224)), transforms.ToTensor()]
        op_train = self.augmentation_list + [transforms.Resize((224, 224)), transforms.ToTensor()]
        
        if self.normalize:
            op_test.append(normalize)
            op_train.append(normalize)
            
        return transforms.Compose(op_train), transforms.Compose(op_test)
    
    def to_binary(self, label_list):
        return [torch.tensor(label).bool().float() for label in label_list]

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T50(
                img_dir=os.path.join(self.dataset_dir, 'videos', video), 
                label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                transform=transform,
                target_transform=self.target_transform
            )
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(
                img_dir=os.path.join(self.dataset_dir, 'videos', video), 
                label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)),
                transform=transform,
                target_transform=self.target_transform
            )
            iterable_dataset.append(dataset)
        self.val_dataset = ConcatDataset(iterable_dataset)

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(
                img_dir=os.path.join(self.dataset_dir, 'videos', video), 
                label_file=os.path.join(self.dataset_dir, 'labels', '{}.json'.format(video)), 
                transform=transform,
                target_transform=self.target_transform
            )
            iterable_dataset.append(dataset)
        self.test_dataset = ConcatDataset(iterable_dataset)
        
    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)

class T50(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_transform=None):
        with open(label_file, "rb") as f:
            label_data = json.load(f)
        self.label_data = label_data["annotations"]
        self.frames = list(self.label_data.keys())
        self.img_dir = img_dir
        self.targets = [self.label_data[frame] for frame in self.frames]
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.frames)

    def get_binary_labels(self, labels):
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([100])
        
        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0: triplet_label[triplet[0]] += 1
            
            tool = label[1:7]
            if tool[0] != -1.0: tool_label[tool[0]] += 1
            
            verb = label[7:8]
            if verb[0] != -1.0: verb_label[verb[0]] += 1
            
            target = label[8:14]  
            if target[0] != -1.0: target_label[target[0]] += 1       
            
            phase = label[14:15]
            if phase[0] != -1.0: phase_label[phase[0]] += 1
            
        return (triplet_label, tool_label, verb_label, target_label, phase_label)
    
    def __getitem__(self, index):
        labels = self.targets[index]
        basename = "{}.png".format(str(self.frames[index]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        labels = self.get_binary_labels(labels)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels