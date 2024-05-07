import pandas as pd
import numpy as np

if __name__ == "__main__":
    df=pd.read_csv(r"./depth_gt/results_Marigold025.csv")

    print('M025')
    mean_value=df['PSNR'].mean()
    print('Mean Value PSNR: '+str(mean_value))
    median_value=df['PSNR'].median()
    print('Median Value: '+str(median_value))
    std_value=df['PSNR'].std()
    print('Standard Deviation Value: '+str(std_value))
    
    print("-----")
    mean_value=df['SSIM'].mean()
    print('Mean Value SSIM: '+str(mean_value))
    median_value=df['SSIM'].median()
    print('Median Value: '+str(median_value))
    std_value=df['SSIM'].std()
    print('Standard Deviation Value: '+str(std_value))

    print("-----")
    mean_value=df['LPIPS'].mean()
    print('Mean Value LPIPS: '+str(mean_value))
    median_value=df['LPIPS'].median()
    print('Median Value: '+str(median_value))
    std_value=df['LPIPS'].std()
    print('Standard Deviation Value: '+str(std_value))

