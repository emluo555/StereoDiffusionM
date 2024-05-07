from PIL import Image
import numpy as np
import sys
import argparse
from glob import glob
import os
from tqdm import tqdm, trange

import torch

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--img_path', metavar='path', required=True,
                        help='file path')
    parser.add_argument('--output_path_l', metavar='path', required=True,
                        help='output path')
    parser.add_argument('--output_path_r', metavar='path', required=True,
                        help='output path')

    args = parser.parse_args()

    rgb_filename_list = glob(os.path.join(args.img_path, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    print('started?')

    rgb_filename_list = sorted(rgb_filename_list)
    for rgb_path in tqdm(rgb_filename_list, desc="cutting in half", leave=True):
        image = np.array(Image.open(rgb_path))[:, :, :3]
        left = image[:,:512]
        right = image[:,512:]

        img_name_base = os.path.splitext(os.path.basename(rgb_path))[0] 
        save_path_left = os.path.join(args.output_path_l, f'{img_name_base}_l.png')

        save_path_right = os.path.join(args.output_path_r, f'{img_name_base}_r.png')
        Image.fromarray(left).save(save_path_left)
        Image.fromarray(right).save(save_path_right)


    