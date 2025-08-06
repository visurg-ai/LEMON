import argparse
import os
import pathlib
import tqdm
import gc
import torch.utils.tensorboard
import cv2
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch
import torchvision
import pickle
import torch.nn.functional as F
from functools import partial
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor, Lambda
from typing import Any, Callable, Optional, Tuple
import PIL
import json
import numpy as np
from ultralytics import YOLO



def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the  dataset (required: True)',
        '-j': 'Path to the root of the  dataset json (required: True)',
        '-o': 'Path to the processed video output of the  dataset (required: True)',
        '--classify-models': 'Path to the .pt models file for frame classifier (required: True)',
        '--segment-models': 'Path to the .pt models file for uninfor part segmentation (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description=' dataset video classifier')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))
    parser.add_argument('-j', '--input-json', required=True, type=str, 
                        help=help('-j'))
    parser.add_argument('-o', '--output', required=True, type=str, 
                        help=help('-o'))
    parser.add_argument('--classify-models', required=True, type=str, help=help('--classify-models'))
    parser.add_argument('--segment-models', required=True, type=str, help=help('--segment-models'))

    # Read parameters
    args = parser.parse_args()
    
    return args



class TestDataset(VisionDataset):
    def __init__(
        self,
        images: list,
        transform: Optional[Callable] = None,
        size: int = 384,
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



def cut_video(arr, frame_interval):
    min_consecutive_frames = 3 * frame_interval

    informative_indices = np.where(arr != 1)[0]


    if len(informative_indices) < min_consecutive_frames:
        print(f"ERROR, continuous informative frames less than {min_consecutive_frames}")
        return None, None, None

    start_index = 0
    for i in range(len(informative_indices) - min_consecutive_frames + 1):

        if informative_indices[i + min_consecutive_frames - 1] - informative_indices[i] == min_consecutive_frames - 1:
            start_index = informative_indices[i]
            break


    end_index = len(arr) - 1
    for i in range(len(informative_indices) - 1, min_consecutive_frames - 2, -1):
        if informative_indices[i] - informative_indices[i - min_consecutive_frames + 1] == min_consecutive_frames - 1:
            end_index = informative_indices[i]
            break

    processed_arr = arr[start_index:end_index + 1]

    return processed_arr, start_index, end_index



def extract_frames(video_path):
    print("start extracting frames")
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = int(fps)  # Convert the fps to an integer
    count = 0
    number = 0
    video_file = video_path.split('/')[-1]
    video_id = video_file.split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        count += 1

    cap.release
    
    return frames



def non_max_suppression(bboxes, scores, iou_threshold):

    selected_indices = []
    
    # Sort bounding boxes by confidence score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    while len(sorted_indices) > 0:
        # Select the bounding box with the highest confidence score
        best_idx = sorted_indices[0]
        selected_indices.append(best_idx)
        
        # Calculate IoU with other bounding boxes
        ious = [calculate_iou(bboxes[best_idx], bboxes[idx]) for idx in sorted_indices[1:]]
        
        # Filter out bounding boxes with high IoU
        filtered_indices = [idx for idx, iou in zip(sorted_indices[1:], ious) if iou <= iou_threshold]
        sorted_indices = filtered_indices
    
    return selected_indices

def calculate_iou(bbox1, bbox2):

    # Calculate intersection area
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    # Calculate union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def build_preprocessing_transforms(size: int = 384, randaug_n: int = 2, 
                                   randaug_m: int = 14):

    
    # Preprocessing for testing
    valid_preproc_tf = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((size,size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4299694, 0.29676908, 0.27707579), (0.24373249, 0.20208984, 0.19319402)),#frame_surgNonsurg
    ])

    return valid_preproc_tf

def build_model(nclasses: int = 2, mode: str = None, segment_model: str = None):

    if mode == 'classify':
        #net of Resnet18
        net = torchvision.models.resnet18(num_classes = nclasses)
        net.cuda()
    if mode == 'mask':
        net = YOLO(segment_model)

    return net

def validate_cmdline_params(args):
    # Input directory must exist
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')
    return args


def process_frames(path: str, informative_videos:list, test_preproc_tf = None, classify_models = None, segment_models = None, output_dir = None, classes = None, size: int = 384, threshold = 0.7):

    # If it is a file, and the MD5 has not been already computed
    if os.path.isfile(path):
        # Summarise video
        filename, file_extension = os.path.splitext(path)
        file_name = path.split('/')[-1]
        file_id = file_name.split('.')[0]
        if file_extension == '.mp4' and path.split('/')[-1] in informative_videos:
            bounding_box_results = {}
            cap = cv2.VideoCapture(path)
            # Check if the video file is opened successfully
            if not cap.isOpened():
                print("Error: Could not open video file.")
                exit()
            
            # Get video properties
            frame_interval = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release

            # Create a VideoWriter object to write the processed frames to a new video file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose a different codec based on your needs
            out = cv2.VideoWriter(os.path.join(output_dir, file_name), fourcc, frame_interval, (frame_width, frame_height))


            predicted_results = {}
            # Extract frames from the current video
            images = extract_frames(path)
            # Create PyTorch dataset for the frames
            dataloader = load_dataset(images=images, test_preproc_tf=test_preproc_tf, bs=1024)
            print(f"Finish data loading of {file_name}...")
    
            # Iterate over frames and classify them
            with torch.no_grad():
                # Create progress bar
                result = np.empty((0,))
                # Loop over testing data points
                for batch_idx, inputs in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                    # Perform inference
                    print(f"start frame classifying and video trimming {file_name}...")
                    inputs = inputs.to('cuda')
                    outputs = [classify_models[i](inputs) for i in range(len(classify_models))]
                    output_torch = torch.argmax(sum(outputs)/len(outputs), dim = 1)
                    predicted_array = output_torch.detach().cpu().numpy()
                    result = np.concatenate((result, predicted_array))
                result = result.flatten()
                cutted_result, start_index, end_index = cut_video(result, frame_interval)
                cutted_images = images[start_index: end_index + 1]

            print(f"start frame preprocessing {file_name}...")
            for num, value in tqdm.tqdm(enumerate(cutted_result), total=len(cutted_result)):
                if value == 0:
                    boxes = []
                    confs = []
                    selected_indices = []
                    results = [segment_models[i](cutted_images[num], verbose=False)[0] for i in range(len(segment_models))]
                    for i in range(len(results)):
                        boxes.extend(results[i].boxes.xyxy.tolist())
                        confs.extend(results[i].boxes.conf.tolist())
                    #print(boxes,boxes[0],'list:type(boxes[0])','float:type(boxes[0][0])','boxes')
                    if len(boxes) > 1:
                        selected_indices = non_max_suppression(boxes, confs, threshold)
                        #print(selected_indices, boxes[0][1])
                        # Make the detected region black in the original image
                        selected_boxes = []
                        for ind in selected_indices:
                            cutted_images[num][int(boxes[ind][1]):int(boxes[ind][3]), int(boxes[ind][0]):int(boxes[ind][2]), :] = 0
                            selected_boxes.append(boxes[ind])
                        bounding_box_results[f'{file_id}-{num}']= {'class':'surgical','boxes':selected_boxes}
                    elif len(boxes) == 1:
                        cutted_images[num][int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2]), :] = 0
                        bounding_box_results[f'{file_id}-{num}']= {'class':'surgical','boxes':boxes}
                    else:
                        bounding_box_results[f'{file_id}-{num}']= {'class':'surgical','boxes':None}
                else:
                    cutted_images[num][:, :, :] = 0
                    bounding_box_results[f'{file_id}-{num}']= {'class':'non-surgical','boxes':None}

                out.write(cutted_images[num])

            with open(f'{output_dir}/{file_id}.json', 'w') as gt_file:
                json.dump(bounding_box_results, gt_file, indent=4)
            out.release()
            
        else:
            print(f'{path} is not an informative video, discarded')
            
    else:
        raise ValueError('[ERROR] ' + path + ' is not a file')


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    with open(args.input_json, 'r') as file_videos:
        videos = json.load(file_videos)
    informative_videos = [video['youtubeId']+'.mp4' for video in videos]
    test_preproc_tf = build_preprocessing_transforms()

    classes = ('surgical', 'non-surgical')
    num_classes = len(classes)
    # Build model, Load weights from file
    cla_model = os.listdir(args.classify_models)
    classify_models = []
    for i in range(len(cla_model)):
        model_path = os.path.join(args.classify_models,cla_model[i])
        net = build_model(nclasses=num_classes, mode='classify')
        # Enable multi-GPU support
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        state = torch.load(model_path, map_location=torch.device('cuda'))
        net.load_state_dict(state['net'])
        net.eval()
        classify_models.append(net)
    seg_model = os.listdir(args.segment_models)
    segment_models = []
    for i in range(len(seg_model)):
        model_path = os.path.join(args.segment_models,seg_model[i])
        segment_models.append(build_model(mode='mask', segment_model = model_path))
                          
    all_videos = os.listdir(args.input)
    print(f'all videos: {len(all_videos)}')
    processed_videos = os.listdir(args.output)
    unprocessed_videos = [file for file in all_videos if file.endswith('.mp4') and file.replace('.mp4','.json') not in processed_videos]
    print(f'unprocessed video: {len(unprocessed_videos)}')
    for vid in unprocessed_videos:
        process_frames(path=os.path.join(args.input,vid), informative_videos=informative_videos,test_preproc_tf=test_preproc_tf,classify_models=classify_models,segment_models=segment_models,output_dir=args.output,classes=classes)


if __name__ == '__main__':
    main()
