import argparse
import html
import json
import os
import random
import re
from functools import partial
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
import time

def scan_recursively(root):
    num = 0
    for entry in os.scandir(root):
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            num += 1
            if num % 100 == 0:
                print(f"Scanned {num} directories.")
            yield from scan_recursively(entry.path)

def get_filelist(file_path, exts=None):
    filelist = []
    time_start = time.time()

    # == OS Walk ==
    # for home, dirs, files in os.walk(file_path):
    #     for filename in files:
    #         ext = os.path.splitext(filename)[-1].lower()
    #         if exts is None or ext in exts:
    #             filelist.append(os.path.join(home, filename))

    # == Scandir ==
    obj = scan_recursively(file_path)
    for entry in obj:
        if entry.is_file():
            ext = os.path.splitext(entry.name)[-1].lower()
            if exts is None or ext in exts:
                filelist.append(entry.path)

    time_end = time.time()
    print(f"Scanned {len(filelist)} files in {time_end - time_start:.2f} seconds.")
    return filelist

def split_by_capital(name):
    # BoxingPunchingBag -> Boxing Punching Bag
    new_name = ""
    for i in range(len(name)):
        if name[i].isupper() and i != 0:
            new_name += " "
        new_name += name[i]
    return new_name
def process_ucf101(root, split):
    root = os.path.expanduser(root)
    video_lists = get_filelist(os.path.join(root, split))
    classes = [x.split("/")[-2] for x in video_lists]
    classes = [split_by_capital(x) for x in classes]
    samples = list(zip(video_lists, classes))
    output = f"ucf101_{split}.csv"

    df = pd.DataFrame(samples, columns=["path", "text"])
    df.to_csv(output, index=False)
    print(f"Saved {len(samples)} samples to {output}.")
process_ucf101("/home/v-zongyili/svd/generative-models/ori_tats/TATS/data/datasets/ucf101","train")