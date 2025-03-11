"""
@brief  Script to download a number of videos from Youtube.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   21 July 2021.
"""

import argparse
import os
import string
import pathlib
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import PIL
import json
import numpy as np
import multiprocessing

# My imports
import you2dl

def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-a': 'Youtube channel name',
        '-c': 'Path to the cookies file, if you are tying to download content \
               that requires login, use this option to pass your cookies',
        '-h': 'Display help',
        '-k': 'Search keywords',
        '-n': 'Number of videos to download',
        '-o': 'Path to the output folder',
        '-e': 'Path to a directory containing previously downloaded videos \
               that must not be downloaded again',
        '-s': 'Skip video download',
        '-d': 'Additionally download description'
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    parser = argparse.ArgumentParser(description='Youtube video downloader.')
    parser.add_argument('-a', '--author', default=None, type=str, 
                        help=help('-a'))
    parser.add_argument('-c', '--cookies', default=None, type=str,
                        help=help('-c'))
    parser.add_argument('-k', '--keyword', default=None, type=str, 
                        help=help('-k'))
    parser.add_argument('-n', '--number', default=None, type=int, 
                        help=help('-n'))
    parser.add_argument('-o', '--output', default=None, type=str, 
                        help=help('-o'))
    parser.add_argument('-e', '--existing', default=None, type=str, 
                        help=help('-e'))
    parser.add_argument('-s', '--skip-video', action='store_true',
                        help=help('-d'))
    parser.add_argument('-d', '--description', action='store_true',
                        help=help('-d'))
    parser.add_argument('--video-path', default=None)
    parser.add_argument('--without-audio', action='store_true')
    parser.add_argument('--audio-separated', action='store_true')
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('-i', '--ignore-words', default=None, type=str)
    args = parser.parse_args()
    
    # Convert parameters to the proper datatypes and data structures
    if args.keyword is not None:
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    if args.ignore_words is not None:
        args.ignore_words = [m.strip() for m in args.ignore_words.split(',')]
    if args.output is None:
        raise ValueError("[ERROR] An output folder has not been provided.")
    
    return args


def main():
    # Read command line parameters
    args = parse_cmdline_params()

    # Create destination folder
    try:
        os.mkdir(args.output)
    except FileExistsError:
        raise ValueError('[ERROR] Output directory already exists.')

    # Collect list of already downloaded videos
    already_downloaded = []
    if args.existing is not None:
        already_downloaded = you2dl.find_all_videos_byjson(args.existing)
    
    if args.video_path is not None:
        video_urls = you2dl.find_all_videos_byjson(args.video_path)
    else:
        # Run Youtube search
        video_urls = you2dl.search(args.keyword, args.author, args.number, args.ignore_words)

    # Prune video urls, removing those videos that we have already downloaded
    if already_downloaded:
        video_urls = you2dl.prune_video_list(video_urls, already_downloaded)

    print('[INFO] Videos to be downloaded:', len(video_urls))

    # Download list of Youtube videos
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = [pool.apply_async(you2dl.download, 
        args=(url, args.output, args.cookies, 3, args.skip_video, not args.description, args.without_audio, args.audio_separated, args.audio_only)) for url in video_urls]
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
