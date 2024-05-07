import torch
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import cv2
import argparse, os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from glob import glob
from tqdm import tqdm
import csv



EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
#LPIPS Functions
def _norm_depth(depth,max_val=1):
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)
    return out

def preprocess(gt_path, imgR_path):
    imageGT=Image.open(gt_path)
    imageR=Image.open(imgR_path)
    imgGTRight = np.asarray(imageGT)
    imgRight= np.asarray(imageR)

    normalGT =  np.transpose(np.expand_dims(_norm_depth(imgGTRight), axis=0), (0, 3, 1, 2)).astype(np.single)
    normalRight =  np.transpose(np.expand_dims(_norm_depth(imgRight), axis=0), (0, 3, 1, 2)).astype(np.single)
    
    normalGT = torch.from_numpy(normalGT)
    normalRight = torch.from_numpy(normalRight)

    return normalGT, normalRight

def run_LPIPS(gt_path, imgR_path):
    imgGT, imgR = preprocess(gt_path, imgR_path)
    _ = torch.manual_seed(123)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True)
    score = lpips(imgGT, imgR)
    return score.item()

#SSIM functions
def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret

def run_SSIM(gt_path, imgR_path):
    imgGT, imgR = preprocess(gt_path, imgR_path)
    score = ssim(imgGT, imgR, val_range=1)
    return score.item()

#PSNR
def run_PSNR(path_imgGT, path_imgR):
    original = cv2.imread(path_imgGT)
    compressed = cv2.imread(path_imgR)

    value = cv2.PSNR(original, compressed) 
    return value

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--gt_path', metavar='path', required=True,
                        help='gt file path')
    parser.add_argument('--imgR_path', metavar='path', required=True,
                        help='generated img path')
    parser.add_argument('--method', metavar='path', required=False,
                        help='method')


    args = parser.parse_args()

    rgb_filename_list = glob(os.path.join(args.gt_path, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)

    for rgb_path in tqdm(rgb_filename_list, desc="running the tests", leave=True):
        img_name_base = os.path.splitext(os.path.basename(rgb_path))[0] 
        img_name_base_r = img_name_base.split("_c")[0]
       # print(img_name_base, img_name_base_r)

        gt_path = os.path.join(args.gt_path, f'{img_name_base}.png')
        imgR_path = os.path.join(args.imgR_path, f'{img_name_base_r}_05.png')

        valuePSNR = run_PSNR(gt_path, imgR_path)
        valueSSIM = run_SSIM(gt_path, imgR_path)
        valueLPIPS = run_LPIPS(gt_path, imgR_path)

        with open('./combine/results_Marigold05.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([f'{img_name_base}', valuePSNR,valueSSIM,valueLPIPS])


