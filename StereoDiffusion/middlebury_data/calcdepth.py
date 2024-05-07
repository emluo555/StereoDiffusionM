import numpy as np
import re
import sys
from readpfm import read_pfm
import argparse
from PIL import Image
import torch
import os

def load_512(image, left=0, right=0, top=0, bottom=0):

    h, w= image.shape
    if w != h:
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
    print('img ',image.shape)
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def calc_depth(d,baseline,focal):
    fact = baseline*focal
    numerator = np.full(d.shape,fact)
    mask = ~np.isinf(d)
    depth = np.where(mask, numerator / d, d)
    return depth / 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--file', metavar='path', required=True,
                        help='file path')
    parser.add_argument('--filename', metavar='path', required=True,
                        help='file path')
    parser.add_argument('--baseline', required=True,type=float,
                        help='baseline')
    parser.add_argument('--focal', required=True,type=float,
                        help='focal')

    args = parser.parse_args()
    
    img_name_base = args.filename

    d = read_pfm(args.file)[0]
    depth = calc_depth(d,args.baseline,args.focal)
    print(depth)
    infindices = np.isinf(depth)
    print(infindices,infindices.shape,np.count_nonzero(infindices))
    depth[infindices]=0
    print(depth.min(),depth.max())


    depth_resize = load_512(depth)
   # depth_resize = np.nan_to_num(depth_resize, nan=0.0)
    depth_resize[depth_resize<0]=0

    tensor= torch.tensor(depth_resize).unsqueeze(0)
    
    print(tensor)
    print(tensor.shape, depth_resize.min(),depth_resize.max())
    torch.save(tensor, f"./Marigold/output/tensor_gt/{img_name_base}.pt")


    #print(depth_resize, depth_resize.shape, depth_resize, depth_resize)