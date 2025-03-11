import os
import json
import lmdb
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
import random
from copy import deepcopy
import torch.utils.tensorboard
import torchvision
import gc
from multiprocessing import Pool
import torch.multiprocessing as mp
from functools import partial
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor, Lambda
from typing import Any, Callable, Optional, Tuple
import cv2
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestDataset(VisionDataset):
    def __init__(
        self,
        images: list,
        transform: Optional[Callable] = None,
        size: int = 448,
    ) -> None:

        self.images = images
        self.transform=transform
        self.size = size
        self.data = []
        for image in images:
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_rgb = cv2.resize(im_rgb, (size,size))
            self.data.append(im_rgb)

    def __getitem__(self, index: int):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)



def load_dataset(images, test_preproc_tf, bs: int, num_workers: int = 2):
    # Load testing set
    test_ds = TestDataset(images=images, transform=test_preproc_tf)
    # Create dataloader
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, 
                                          shuffle=False, 
                                          num_workers=num_workers)

    return test_dl


def remove_all_black_borders(image_np):
    """
    Remove all-black borders from an image.
    
    Parameters:
        image_np (numpy.ndarray): The input image as a NumPy array.
    
    Returns:
        cropped_image_np (numpy.ndarray): The cropped image with black borders removed.
    """
    def find_non_black_bounds(image):
        """
        Find the bounds of the non-black area in the image.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find all rows and columns that are not completely black
        non_black_rows = np.where(np.any(gray > 0, axis=1))[0]
        non_black_cols = np.where(np.any(gray > 0, axis=0))[0]

        if non_black_rows.size == 0 or non_black_cols.size == 0:
            return 0, image.shape[0], 0, image.shape[1]  # Entire image is black

        # Get bounds of the non-black area
        top = non_black_rows[0]
        bottom = non_black_rows[-1] + 1
        left = non_black_cols[0]
        right = non_black_cols[-1] + 1
        
        return top, bottom, left, right

    top, bottom, left, right = find_non_black_bounds(image_np)
    cropped_image_np = image_np[top:bottom, left:right]

    return cropped_image_np


def remove_black_borders(image_np, threshold_ratio=0.35):
    """
    Remove borders from an image that have a high ratio of black pixels.
    
    Parameters:
        image_np (numpy.ndarray): The input image as a NumPy array.
        threshold_ratio (float): The threshold ratio of black pixels to consider a row/column as black.
                                 Default is 0.1 (i.e., 10%).
    
    Returns:
        cropped_image_np (numpy.ndarray): The cropped image with black borders removed.
    """
    def find_non_black_bounds(image, threshold_ratio):
        """
        Find the bounds of the non-black area in the image based on the threshold ratio of black pixels.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to binary (black and white)
        #_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find all rows and columns that are not considered black
        row_black_counts = np.sum(gray == 0, axis=1)
        col_black_counts = np.sum(gray == 0, axis=0)
        
        non_black_rows = np.where(row_black_counts <= threshold_ratio * gray.shape[1])[0]
        non_black_cols = np.where(col_black_counts <= threshold_ratio * gray.shape[0])[0]
        
        if non_black_rows.size == 0 or non_black_cols.size == 0:
            return 0, image.shape[0], 0, image.shape[1]  # Entire image is black or exceeds the threshold

        # Get bounds of the non-black area
        top = non_black_rows[0]
        bottom = non_black_rows[-1] + 1
        left = non_black_cols[0]
        right = non_black_cols[-1] + 1
        
        return top, bottom, left, right

    top, bottom, left, right = find_non_black_bounds(image_np, threshold_ratio)
    # Ensure the cropping bounds are valid
    if bottom - top > 100 and right - left > 50 and top >= 0 and left >= 0 and bottom <= image_np.shape[0] and right <= image_np.shape[1]:
        cropped_image_np = image_np[top:bottom, left:right]
    else:
        cropped_image_np = None

    return cropped_image_np



def extract_frames(video_path):
    print("start extracting frames")
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    number = 0
    video_file = video_path.split('/')[-1]
    video_id = video_file.split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        #cv2.imwrite(f'{output}/predicted_frames/{video_id}_{number}.jpg', frame)
        number += 1
        count += 1

    cap.release
    
    return frames

def create_lmdb(video_folder, output_json, lmdb_path, map_size, model_list = None, test_preproc_tf = None, classes = ('surgical', 'non-surgical'), image_size = (640,640)):
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    video_list = [video for video in os.listdir(video_folder) if video.endswith('.mp4')]
    print(f'all videos are {len(video_list)}')
    predicted_results = {}
    if os.path.isfile(output_json):
        with open(output_json, 'r') as json_o:
            predicted_results = json.load(json_o)
            extracted_vids = set(predicted_results.keys())
            video_list = [vii for vii in video_list if vii.split('.')[0] not in extracted_vids]

    print(f'rest videos are {len(video_list)}')

    
    try:
        txn = env.begin(write=True)
        for index_vid, video in enumerate(tqdm(video_list, desc="Storing videos in LMDB")):    
            video_id = video.split('.')[0] 
            images = extract_frames(os.path.join(video_folder, video))
            #dataloader = load_dataset(images=images, test_preproc_tf=test_preproc_tf, bs=10000)
            print("Finish data loading...")
            
            json_file = os.path.join(video_folder, video.replace('.mp4', '.json'))
            with open(json_file, 'r') as json_f:
                json_data = json.load(json_f)
            
            if len(json_data) != len(images):
                print(f"ERROR, the length of json {len(json_data)} is not equal to frames' length {len(images)}")
                predicted_results[f'{video_id}'] = 'not align'
                continue
            if len(json_data) == len(images):
                predicted_results[f'{video_id}'] = 'align'
            
            with open(output_json, 'w') as gt_file:
                json.dump(predicted_results, gt_file, indent=4)
            
            print(f"start storing {video_id} into lmdb...")
            for num, json_frame in enumerate(tqdm(json_data, desc="Storing images in LMDB")):
                if json_data[json_frame]['class'] == 'surgical':
                    img = remove_all_black_borders(images[num])
                    img = remove_black_borders(img, threshold_ratio=0.35)
                    
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get the original dimensions
                    height, width, _ = img.shape

                    # Calculate the new dimensions after scaling by 0.8
                    new_height = int(height * 0.5)
                    new_width = int(width * 0.5)
                    
                    img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
                    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    
                    img_bytes = img_encoded.tobytes()
                    #img_np = np.array(img, dtype=np.uint8)
                    #img_bytes = img_np.tobytes()
                    img_key = f'{video_id}_{num}'.encode('utf-8')
                    txn.put(img_key, img_bytes)
                



            # Commit any remaining data
            if index_vid + 1 < len(video_list):
                txn.commit()
                txn = env.begin(write=True)
            #if index_vid +1 == len(video_list):
                #txn.commit()

    except Exception as e:
        print(f"Exception occurred: {e}")
    
    finally:
        txn.commit()
        env.close()
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create LMDB for image dataset")
    parser.add_argument('--video-folder', type=str, required=True, help="Path to the image folder")
    parser.add_argument('--output-json', type=str, default=None, required=False, help="Path to the label JSON file")
    parser.add_argument('--lmdb-path', type=str, required=True, help="Path to the output LMDB file")
    parser.add_argument('--map-size', type=float, default=1e13, help="Map size for LMDB")


    args = parser.parse_args()

    create_lmdb(args.video_folder, args.output_json, args.lmdb_path, map_size=args.map_size)
