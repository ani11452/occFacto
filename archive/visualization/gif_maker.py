import glob
from argparse import ArgumentParser
import numpy as np
import imageio
import cv2
def make_gif():
    image = []
    for j in range(50):
        filename = f"~/workspace/websites/diffFacto.github.io/videos/combined.gif"
        image.append(imageio.imread(filename))
    imageio.mimsave(f'/Users/georgenakayama/Desktop/chair_mixing/spiral.gif', image, format='GIF', duration=50, loop=0)

def img2mse(x, y, reduce: bool=True):
    l = (x - y) ** 2
    if reduce:
        return np.mean(l)
    return l

def mse2psnr(x):
    return -10. * np.log(x) / np.log(np.array([10.]))

if __name__ == "__main__":
    
    im1 = imageio.imread("~/Desktop/img_199000_8c794d3b6b3b95d39c84.png") / 255.
    im2 = imageio.imread("~/Desktop/img_200001_2ebeaa52b3485e73ad6a.png") / 255.
    gt_im = imageio.imread("~/Desktop/img_gt_199000_35fb42d9802b90c2e7cb.png") / 255.
    psnr1 = mse2psnr(img2mse(im1, gt_im))
    psnr2 = mse2psnr(img2mse(im2, gt_im))
    print(psnr1, psnr2, img2mse(im1, gt_im), img2mse(im2, gt_im), img2mse(im2, im1))
    
