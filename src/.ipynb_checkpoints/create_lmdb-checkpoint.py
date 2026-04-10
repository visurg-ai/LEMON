import os
import json
import lmdb
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def create_lmdb(image_folder, label_json, lmdb_path, map_size=1e13, image_size=(224, 224)):
    """
    Creates an LMDB database from a folder of images and a JSON label file.
    
    Args:
        image_folder (str): Directory containing the images.
        label_json (str): Path to a JSON file mapping filenames to labels.
                          Format: {"image1.jpg": "class_name", ...}
        lmdb_path (str): Output path for the LMDB folder.
        map_size (int): Maximum size of the LMDB database in bytes.
        image_size (tuple): Target size (width, height) to resize images.
    """
    
    # Load labels
    print(f"Loading labels from {label_json}...")
    with open(label_json, 'r') as f:
        labels = json.load(f)

    # Open LMDB environment
    # map_size should be significantly larger than the expected dataset size
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    with env.begin(write=True) as txn:
        for img_name, label in tqdm(labels.items(), desc="Storing images"):
            try:
                img_path = os.path.join(image_folder, img_name)
                
                # Check if file exists
                if not os.path.exists(img_path):
                    print(f"Warning: File {img_path} not found. Skipping.")
                    continue

                # Read and Process Image
                # OpenCV loads in BGR, convert to RGB for consistency with PyTorch/PIL
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}. Skipping.")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                
                # Convert to raw bytes
                # Note: Storing raw numpy bytes is fast but takes more space than encoded JPEGs.
                # If space is an issue, consider cv2.imencode within the loop.
                img_bytes = img.tobytes()
                
                # Handle label encoding (ensure it's a string)
                label_str = str(label)
                label_bytes = label_str.encode('utf-8')
                
                # Create Keys
                img_key = f'img_{img_name}'.encode('utf-8')
                label_key = f'label_{img_name}'.encode('utf-8')
                
                # Write to LMDB
                txn.put(img_key, img_bytes)
                txn.put(label_key, label_bytes)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    env.close()
    print(f"Finished creating LMDB at {lmdb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LMDB dataset from images and a JSON label file.")
    
    parser.add_argument('--image-folder', type=str, required=True, 
                        help="Path to the folder containing images.")
    parser.add_argument('--label-json', type=str, required=True, 
                        help="Path to the JSON file mapping 'filename' -> 'label'.")
    parser.add_argument('--lmdb-path', type=str, required=True, 
                        help="Destination path for the LMDB output.")
    parser.add_argument('--map-size', type=float, default=1e12, 
                        help="Maximum map size for LMDB in bytes (default: 1TB). Increase if dataset is large.")
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224], 
                        help="Resize images to (width, height). Default: 224 224")

    args = parser.parse_args()
    
    create_lmdb(
        image_folder=args.image_folder, 
        label_json=args.label_json, 
        lmdb_path=args.lmdb_path, 
        map_size=args.map_size, 
        image_size=tuple(args.image_size)
    )