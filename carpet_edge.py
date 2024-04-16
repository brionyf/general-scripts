import numpy as np
import cv2
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import os
from skimage.filters.rank import median

DEBUG = False
BASE_PATH = os.getcwd()

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory to load/save images from/to")
    return parser

def get_image_filenames(folder):
    return sorted([folder + file_name for file_name in os.listdir(folder) if '.png' in file_name and 'cropped' not in file_name and 'marked' not in file_name])

if __name__ == '__main__':

    args = get_parser().parse_args()

    images = get_image_filenames(args.img_dir)

    for filename in images:

        img = cv2.imread(filename)#, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        ratio = 0.2
        img = cv2.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))

        # cropped = img[:, 100:]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur_k = 9 # kernel
        edge_k = 5 # kernel
        # footprint = np.ones((img.shape[0],1), np.uint8)
        footprint = np.ones((100,1), np.uint8)
        img_blur = cv2.GaussianBlur(img_gray, (blur_k, blur_k), sigmaX=0, sigmaY=0)
        # edges = cv2.Canny(img, 100, 150)
        edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=edge_k)
        edges_norm = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)#.astype(np.uint8), CV_16S, CV_8U
        edges = median(edges_norm, footprint)
        edges = cv2.Canny(edges, 50, 200)

        img[edges>250] = (0,0,255) #[edges==255]

        base_crop = 100
        padding = 25
        pts = np.argwhere(edges[:, base_crop:] > 0)
        _,x = pts.min(axis=0)

        cv2.imwrite(filename + '_marked', img)
        cv2.imwrite(filename + '_cropped', img[:, (x+base_crop+padding):])

# Command Line:
# conda deactivate
# conda activate detect_env
# cd '/home/brionyf/Desktop/Code/00 generic code'
# python carpet_edge.py --img_dir '/home/brionyf/Desktop/Images/detect carpet edge/'
