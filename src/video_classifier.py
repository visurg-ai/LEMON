"""
@brief   Summarise all the videos of the dataset into a collage. This collage
         will be used to annotate informative/non-informative videos.
@date    19 Dec 2023.
"""

import argparse
import os
import pathlib
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import PIL
import json
import numpy as np
# My imports 
import videosum


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the root of the LEMON dataset (required: True)',
        '--models': 'Path to the .pt models file (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='LEMON dataset video classifier')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))
    parser.add_argument('-o', '--output', required=True, type=str)
    parser.add_argument('--models', required=True, type=str, help=help('--models'))

    # Read parameters
    args = parser.parse_args()
    
    return args

def build_preprocessing_transforms(size: int = 384, randaug_n: int = 2, 
                                   randaug_m: int = 14):
    """
    @brief Preprocessing and data augmentation.

    @param[in]  size  Target size of the images to be resized prior 
                      processing by the network.

    @returns a tuple of two transforms, one for training and another one for testing.
    """
    
    # Preprocessing for testing
    valid_preproc_tf = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((size,size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.44476654, 0.35751768, 0.34613244), (0.28885465, 0.25989645, 0.26113421)),#video
    ])

    return valid_preproc_tf

def build_model(nclasses: int = 2, pretrained: bool = True):
    """
    @brief Create Vision Transformer (ViT) model pre-trained on ImageNet-21k 
           (14 million images, 21,843 classes) at resolution 224x224 
           fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) 
           at resolution 384x384.

    @param[in]  nclasses    Number of classes, CIFAR-10 has obviously 
                            10 classes.
    @param[in]  pretrained  Initialise the model with the pretrained weights
                            of ImageNet-21k and ImageNet 2012.
    """
    #net of Resnet18
    net = torchvision.models.resnet18(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, nclasses)
    net.cuda()

    return net

def validate_cmdline_params(args):
    # Input directory must exist
    if not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input directory does not exist.')
    return args


def classify_videos(result: dict, path: str, nframes: int = 16, width: int = 1920, 
        height: int = 1080, test_preproc_tf=None ,model_list=None, size: int = 384) -> dict:
    """
    @brief Recursive function to computes the summary of each video file in the
           dataset and write it into a file with the same name but
           .png extension.
    @param[in]  path  Path to the root folder of the tree.
    @returns nothing.
    """
    # If it is a file, and the MD5 has not been already computed
    if os.path.isfile(path):
        # Summarise video
        filename, file_extension = os.path.splitext(path)
        
        if file_extension == '.mp4' and not os.path.isfile(filename + '.jpg'):
            try:
                # Compute video summary
                print("Starting Summerising", path)
                vs = videosum.VideoSummariser('time', nframes, width, height)
                im = vs.summarise(path)
                # Write summary image to file
                cv2.imwrite(filename + '.jpg', im, 
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print('[OK] summerization done', path)
               
                with torch.no_grad():
                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im_rgb = cv2.resize(im_rgb, (size,size))
                    if len(im_rgb.shape) != 3:
                        raise ValueError('[ERROR] The input image must be a colour image.')
                    # Preprocess image
                    im_tensor = test_preproc_tf(im_rgb).to('cuda')

                    # Run inference
                    outputs = [model_list[i](torch.unsqueeze(im_tensor, 0)) for i in range(len(model_list))]
                    probabilities = [F.softmax(outputs[m], dim=1) for m in range(len(outputs))]
                    output = torch.argmax(sum(outputs)/len(outputs), dim = 1).item()
                    probability, index = (sum(probabilities)/len(probabilities)).max(dim = 1)

                    # Print result
                    classes = ('informative', 'uninformative')
                    print("{} is a {}!".format(path,classes[output]), output, probability, probability.item(), outputs, probabilities)
                    result[filename.split('/')[-1]] = classes[output]

            except Exception as e:
                # Print information about the exception
                print(f"An exception occurred: {type(e).__name__}: {str(e)}")
                
    elif os.path.isdir(path):
        listing = os.listdir(path) 
        for item in listing:
            result = classify_videos(result = result, path = os.path.join(path, item), test_preproc_tf = test_preproc_tf, model_list = model_list)
    else:
        raise ValueError('[ERROR] ' + path + ' is not a file.')

    return result


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    test_preproc_tf = build_preprocessing_transforms()
    model = os.listdir(args.models)
    model_list = []
    # Build model, Load weights from file
    for i in range(len(model)):
        model_path = os.path.join(args.models,model[i])
        net = build_model()
        # Enable multi-GPU support
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        state = torch.load(model_path, map_location=torch.device('cuda'))
        net.load_state_dict(state['net'])
        net.eval()
        model_list.append(net)
   
    result = {}
    # Compute video summary for all the videos
    result = classify_videos(result = result, path = args.input, test_preproc_tf = test_preproc_tf, model_list = model_list)
    with open(args.output,'w') as result_file:
        json.dump(result, result_file, indent=4)

if __name__ == '__main__':
    main()
