
import argparse, os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
import sys
from typing import Optional, Union, Tuple, List
sys.path.append('./stablediffusion')
sys.path.append('./DPT')
from stablediffusion.ldm.util import instantiate_from_config
from DPT.dpt.models import DPTDepthModel
from stereoutils import *
sys.path.append('./Marigold')
from marigold import MarigoldPipeline
sys.path.append('./prompt-to-prompt')
import ptp_utils 
from skimage.transform import resize
# import p2putil
from diffusers import StableDiffusionPipeline, DDIMScheduler
# torch.set_grad_enabled(False)
import torch.nn.functional as nnf
import abc
import seq_aligner
import shutil
from torch.optim.adam import Adam
import torchvision
import subprocess
from glob import glob
import time






EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True,help="path to image")
    parser.add_argument("--depthmodel_path",type=str,required=False,help='path of depth model')
    parser.add_argument(
        "--deblur",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=9.0,
        help="scale factor of disparity",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["uni", "bi"],
        default="uni"
    )
    #marigold parser
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    return parser.parse_args()

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    if w != h:
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


if __name__ == "__main__":
    
    args = parse_args()
    
    rgb_filename_list = glob(os.path.join(args.img_path, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    start = time.time()
    rgb_filename_list = sorted(rgb_filename_list)[15:16]
    # for rgb_path in tqdm(rgb_filename_list, desc="preprocessing to 512", leave=True):
    #     image  = load_512(rgb_path)
    #     img_name_base = os.path.splitext(os.path.basename(rgb_path))[0] + "_512"
    #     save_path = f"./images_512/{img_name_base}.png"
    #     Image.fromarray(image).save(save_path)
    # print("512 done")
    # subprocess.run(["python","./Marigold/runMT.py","--input_rgb_dir","./images_512", "--output_dir","./Marigold/output",])

    for rgb_path in tqdm(rgb_filename_list, desc="stereo generation", leave=True):
        print('image is ', rgb_path)
        subprocess.run(["python","img2stereoMTF.py","--img_path",rgb_path, "--checkpoint","prs-eth/marigold-v1-0"])
    end = time.time()
    print(end-start)