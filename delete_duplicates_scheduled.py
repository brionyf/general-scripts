# Run commands:
# env: detect_env
# dir: ~/Desktop/Code/00 generic code
# command line: python delete_duplicates.py --img_dir '/home/brionyf/Desktop/Images/ssim_image_similarity'

import cv2
import numpy as np
import os
from glob import glob
import time
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from skimage.metrics import structural_similarity
from tqdm import tqdm
import schedule

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory of folder containing image folders and text file")
    # parser.add_argument("--text_file", type=str, required=True, help="Path of text file containing list of processed folders")
    return parser

def get_folders():
    args = get_parser().parse_args()

    # TODO: get list of folders in cwd
    # all_folders = [x[0] for x in os.walk(args.img_dir)] # also gets sub-subfolders
    # all_folders = glob(args.img_dir+"/*/", recursive = True)
    all_folders = np.array([f.path.replace(args.img_dir,'') for f in os.scandir(args.img_dir) if f.is_dir()])

    # TODO: get list of folders from text file
    with open(Path(args.img_dir,"processed_folders.txt"), "r") as reader:
        # processed_folders = np.array(reader.readlines())
        processed_folders = np.array(reader.read().splitlines())

    # TODO: get list of folders that haven't been processed yet
    # print(all_folders, processed_folders)
    to_process = np.setdiff1d(all_folders, processed_folders)

    if len(to_process) > 0:
        for folder in to_process:
            calc_similarity(args.img_dir + folder)
            print("Removed duplicates from: {}".format(folder))
            with open(Path(args.img_dir,"processed_folders.txt"), "a") as writer:
                writer.write(folder + '\n')
    else:
        current_date = datetime.now().strftime("%d-%m-%Y %H-%M-%S")  # dd/mm/YY H:M:S
        print("There were no new image folders to remove duplicates from today ({})".format(current_date))

def calc_similarity(img_dir):
    file_names = sorted([file_name for file_name in os.listdir(img_dir) if '.png' in file_name and '0_' in file_name])
    image_size = 0.25
    # for i, file in enumerate(file_names):
    print("Original number of images: {}".format(len(file_names)))
    for i in tqdm(range(len(file_names))):
        img = cv2.imread(img_dir + '/' + file_names[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(img.shape[0]*image_size), int(img.shape[1]*image_size)), interpolation = cv2.INTER_AREA)
        if i == 0: compare_img = img
        (ssim_score, _) = structural_similarity(compare_img, img, full=True)#, channel_axis=3)
        # diff = np.mean(abs(compare_img - img))
        # print("Image {} - Diff: {} \tSSIM: {}".format(file, round(diff, 2), round(ssim_score, 2)))
        if (ssim_score > 0.7) and (i != 0):
            os.remove(img_dir + '/' + file_names[i]) # Camera 0
            # os.remove(img_dir + '/1' + file_names[i][1:]) # Camera 1
            # os.remove(img_dir + '/2' + file_names[i][1:]) # Camera 2
        else:
            compare_img = img
    file_names = sorted([file_name for file_name in os.listdir(img_dir) if '.png' in file_name and '0_' in file_name])
    print("Final number of images: {}".format(len(file_names)))

def schedule_test():
    current_date = datetime.now().strftime("%d-%m-%Y %H-%M-%S")  # dd/mm/YY H:M:S
    print("I'm working... {}".format(current_date))

if __name__ == '__main__':

    # schedule.every(1).minutes.do(schedule_test)
    schedule.every(1).minutes.do(get_folders)
    # schedule.every().day.at("15:00").do(get_folders(args))
    # schedule.every().day.at("18:30").do(get_folders(args))
    #schedule.every().day.at("12:42", "Europe/Amsterdam").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
