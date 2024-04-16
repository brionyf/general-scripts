import random
import numpy as np, time, cv2, os
import math
from datetime import datetime
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pywt
# import pywt.data

DEBUG = False
DATETIME = datetime.now().strftime("%d-%m-%Y %H-%M-%S")  # dd/mm/YY H:M:S
BASE_PATH = os.getcwd()
SAVE_PATH = BASE_PATH + '/preprocess ' + DATETIME

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-c", "--crop_size", nargs='+', type=int, default=0, help="Crop size [DEFAULT: 0 0 0 0]")
    # parser.add_argument("-c", "--crop_size", nargs='+', type=float, default=0, help="Crop size ratios [DEFAULT: 0 1.0]")
    parser.add_argument("-r", "--resize_dims", type=int, default=0, help="Resize dimension [DEFAULT: 0 i.e. don't resize]")
    parser.add_argument("-bm", "--binary_mask", type=bool, default=False, help="Create ground truth image? [DEFAULT: False]")
    parser.add_argument("-cm", "--colour_mask", type=bool, default=False, help="Create ground truth instance mask? [DEFAULT: False]")
    parser.add_argument("-t", "--tile_img", type=bool, default=False, help="Crop image into tiles? [DEFAULT: False]")
    parser.add_argument("-pd", "--pad_img", type=bool, default=False, help="Add padding to image? [DEFAULT: False]")
    parser.add_argument("--shads", type=bool, default=False, help="Remove shadows? [DEFAULT: False]")
    parser.add_argument("-he", "--eq_hist", type=bool, default=False, help="Equalise image histogram? [DEFAULT: False]")
    parser.add_argument("-ed", "--erode_dilate", type=bool, default=False, help="Perform erosion and dilation? [DEFAULT: False]")
    parser.add_argument("--draw", type=bool, default=False, help="Draw crop lines on image? [DEFAULT: False]")
    parser.add_argument("--dwt", type=bool, default=False, help="Discrete wavelet transform? [DEFAULT: False]")
    return parser

class Preprocess(object):
    def __init__(self, args):
        if args.crop_size == 0:
            self.crop = False
            self.crop_size = args.crop_size
        else:
            self.crop = True
            self.crop_size = tuple(args.crop_size)
        self.resize_dims = args.resize_dims
        self.mask = args.binary_mask
        self.cmask = args.colour_mask
        self.tile = args.tile_img
        self.padding = args.pad_img
        self.shadows = args.shads
        self.eq_hist = args.eq_hist
        self.erode_dilate = args.erode_dilate
        self.draw_lines = args.draw
        self.dwt = args.dwt

    # def data_augmenter(self, img):
    #     if round(random.random()): img = cv2.flip(img, 0) # 0 to flip up-down
    #     if round(random.random()): img = cv2.flip(img, 1) # 1 to flip left-right
    #     return img

    def dwt_func(self, img): # discrete wavelet transform (edge detection)

        cut_ratios = [0.2, 0.05]
        img = img[int(img.shape[0]*cut_ratios[0]):int(img.shape[0]*(1-cut_ratios[0])),
              int(img.shape[1]*cut_ratios[1]):int(img.shape[1]*(1-cut_ratios[1])), :]
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # dwt_coeffs = pywt.dwt2(img_grey, 'haar')
        dwt_coeffs = pywt.wavedec2(img_grey, 'db1', level=1)
        _, (dwt_horiz, dwt_vert, dwt_diag) = dwt_coeffs
        # _, (dwt_horiz, dwt_vert, dwt_diag), _ = dwt_coeffs

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (dwt_horiz.shape[1],dwt_horiz.shape[0]), interpolation = cv2.INTER_AREA)

        img_tile = Image.new("RGB", (img.shape[1]*2, img.shape[0]*2), "black")
        img_tile.paste(Image.fromarray(img), (0, 0))
        # norm_img = dwt_horiz / np.abs(dwt_horiz).max()
        norm_img = cv2.normalize(np.abs(dwt_horiz), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img_tile.paste(Image.fromarray(norm_img), (img.shape[1], 0))
        # norm_img = dwt_horiz / np.abs(dwt_horiz).max()
        norm_img = cv2.normalize(np.abs(dwt_vert), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img_tile.paste(Image.fromarray(norm_img), (0, img.shape[0]))
        norm_img = cv2.normalize(np.abs(dwt_diag), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img_tile.paste(Image.fromarray(norm_img), (img.shape[1], img.shape[0]))

        return np.asarray(img_tile)

    def erosion_dilation(self, img):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1) # iterations how many times to erode/dilate an image
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def equalise_hist(self, img): # HE is a statistical approach for spreading out intensity values
        adaptive = 1
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if not adaptive:
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # equalize the histogram of the Y channel (brightness)
        else:
            # for background: https://www.tutorialspoint.com/clahe-histogram-equalization-ndash-opencv
            # clipLimit = threshold for contrast limiting, default value is 40
            # tileGridSize = size of grid for histogram equalization, default this is 8×8
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16,16))
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def remove_shadows(self, img):
        rgb_planes = cv2.split(img)
        result_planes = []
        # result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            # norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            # result_norm_planes.append(norm_img)
        # print(np.array(result_planes).shape)
        return np.moveaxis(np.array(result_planes), 0, 2) #, np.array(result_norm_planes)

    def crop_img(self, img): # (x - width, y - height)

        # x_ratio1,x_ratio2 = self.crop_size
        # return img[:, int(img.shape[1]*x_ratio1):int(img.shape[1]*x_ratio2), :]

        y1,x1,y2,x2 = self.crop_size
        if (x1 == x2) or (x2 == 0): x2 = img.shape[1]
        if (y1 == y2) or (y2 == 0): y2 = img.shape[0]
        if DEBUG: print('Crop indices: {}'.format(self.crop_size))
        return img[y1:y2, x1:x2, :]

    def draw_crop_lines(self, img):
        colour = (0,0,255) # BGR
        thickness = 10
        width, height = img.shape[1], img.shape[0]
        # horizontal lines
        cut_ratio = 0.2
        cv2.line(img, (0, int(height*cut_ratio)), (width, int(height*cut_ratio)), colour, thickness)
        cv2.line(img, (0, int(height*(1-cut_ratio))), (width, int(height*(1-cut_ratio))), colour, thickness)
        # vertical lines
        cut_ratio = 0.05
        cv2.line(img, (int(width*cut_ratio), 0), (int(width*cut_ratio), height), colour, thickness)
        cv2.line(img, (int(width*(1-cut_ratio)), 0), (int(width*(1-cut_ratio)), height), colour, thickness)
        return img

    def resize(self, img):
        return cv2.resize(img, (self.resize_dims,self.resize_dims), interpolation = cv2.INTER_AREA)

    def get_binary_mask(self, img):
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros([img.shape[0], img.shape[1]])
        # mask[grey_image == 255] = 255
        return mask

    def get_colour_mask(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print('Image Dimensions : {}'.format(img.shape))
        masked = np.zeros([img.shape[0], img.shape[1], 3])
        colours = ([255,0,0],[255,255,0],[255,0,255],[0,255,0],[255,255,0],[0,0,255])
        for colour in colours:
            mask = img[:,:,:] == colour
            # masked = img.copy()
            masked[mask] = 1
        return masked * img

    def pad_img(self, img):
        h,w,_ = img.shape
        max_dim, min_dim = np.max([h,w]), np.min([h,w])
        # mask = np.zeros([max_dim, max_dim, 3])
        thickness = int((max_dim-min_dim) / 2)
        if h == max_dim:
            padded = cv2.copyMakeBorder(img, 0,0,thickness,thickness, cv2.BORDER_CONSTANT, value=[0,0,0]) # top,bottom,left,right
        elif w == max_dim:
            padded = cv2.copyMakeBorder(img, thickness,thickness,0,0, cv2.BORDER_CONSTANT, value=[0,0,0]) # top,bottom,left,right
        return padded

    def tile_image(self, img, filename):
        # y,x --> rows,cols
        num = (math.ceil(img.shape[0] / self.resize_dims), math.ceil(img.shape[1] / self.resize_dims))
        y_overlap = int(((num[0] * self.resize_dims) - img.shape[0]) / (num[0] - 1))
        x_overlap = int(((num[1] * self.resize_dims) - img.shape[1]) / (num[1] - 1))
        if DEBUG: print('Overlaps: {} and {}'.format(y_overlap, x_overlap))

        tile_count = 0
        for i in range(num[0]):
            y1 = i * (self.resize_dims - y_overlap)
    #         if y1 < 0: y1 = 0
            for j in range(num[1]):
                x1 = j * (self.resize_dims - x_overlap)
    #             if x1 < 0: x1 = 0
                x2, y2 = x1 + self.resize_dims, y1 + self.resize_dims

                if y2 > img.shape[0]:
                    diff = y2 - img.shape[0]
                    y1, y2 = y1 - diff, y2 - diff

                if DEBUG: print(x1, x2, y1, y2)

                new_img = img[y1:y2, x1:x2]

        #         if random.randint(0,1): new_img = cv2.flip(img, 0)
        #         if self.resize_dims != 0: resized = self.resize(new_img)

                tile_count += 1
                # cv2.imwrite(('0000'+str(count))[-4:] + '_' + str(tile_count) + '.png', new_img) # Saving the new image
                # count += 1
                cv2.imwrite(filename.replace('.png','') + '_' + str(tile_count) + '.png', new_img)

        # return count

    def preprocess_images(self):
        file_names = sorted([file_name for file_name in os.listdir(BASE_PATH) if '.jpg' in file_name or '.png' in file_name or '.tiff' in file_name])
        if DEBUG: print(file_names)

        count = 0
        for file_name in file_names:
            # print(file_name)
            img = cv2.imread(BASE_PATH + '/' + file_name, cv2.IMREAD_UNCHANGED)
            if DEBUG: print('Image Dimensions : {}'.format(img.shape))

            if self.crop:
                img = self.crop_img(img)
                if DEBUG: print('Cropped Dimensions : {}'.format(img.shape))
            if self.mask:
                # print("here, mask = {}".format(self.mask))
                img = self.get_binary_mask(img)
                if DEBUG: print('Mask Dimensions : {}'.format(img.shape))
            if self.cmask:
                img = self.get_colour_mask(img)
                if DEBUG: print('Mask Dimensions : {}'.format(img.shape))
            if self.padding:
                img = self.pad_img(img)
                # print('Padded Dimensions : {}'.format(img.shape))
            if (self.resize_dims != 0) and not self.tile:
                img = self.resize(img)
                if DEBUG: print('Resized Dimensions : {}'.format(img.shape))
            if self.shadows:
                img = self.remove_shadows(img)
            if self.eq_hist:
                img = self.equalise_hist(img)
            if self.erode_dilate:
                img = self.erosion_dilation(img)
            if self.draw_lines:
                img = self.draw_crop_lines(img)
            if self.dwt:
                img = self.dwt_func(img)
            if self.tile:
                # count = self.tile_image(img, count)
                self.tile_image(img, file_name)
            else:
                cv2.imwrite(SAVE_PATH + '/' + file_name, img) # Saving the new image

if __name__ == '__main__':

    os.mkdir(SAVE_PATH) #os.path.join(BASE_PATH, 'preprocess', DATETIME))
    os.chdir(SAVE_PATH)

    args = get_parser().parse_args()

    bc = Preprocess(args)
    bc.preprocess_images()
