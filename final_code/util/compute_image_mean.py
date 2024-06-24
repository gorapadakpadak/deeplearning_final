# import argparse
# import numpy as np
# from os.path import join as jpath
# from PIL import Image


# def params():
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--dataroot', type=str, default='datasets/KingsCollege', help='dataset root')
#     parser.add_argument('--height', type=int, default=256, help='image height')
#     parser.add_argument('--width', type=int, default=455, help='image width')
#     parser.add_argument('--save_resized_imgs', action="store_true", default=False, help='save resized train/test images [height, width]')
#     return parser.parse_args()

# args = params()
# dataroot = args.dataroot
# imsize = [args.height, args.width] # (H, W)
# imlist =  np.loadtxt(jpath(dataroot, 'dataset_train.txt'),
#                     dtype=str, delimiter=' ', skiprows=3, usecols=(0))
# mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=np.float64)
# for i, impath in enumerate(imlist):
#     print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
#     image = Image.open(jpath(dataroot, impath)).convert('RGB')
#     image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
#     mean_image += np.array(image).astype(np.float64)

#     # save resized training images
#     if args.save_resized_imgs:
#         image.save(jpath(dataroot, impath))
# print()
# mean_image /= len(imlist)
# Image.fromarray(mean_image.astype(np.uint8)).save(jpath(dataroot, 'mean_image.png'))
# np.save(jpath(dataroot, 'mean_image.npy'), mean_image)

# # save resized test images
# if args.save_resized_imgs:
#     imlist =  np.loadtxt(jpath(dataroot, 'dataset_test.txt'),
#                         dtype=str, delimiter=' ', skiprows=3, usecols=(0))
#     for i, impath in enumerate(imlist):
#         print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
#         image = Image.open(jpath(dataroot, impath)).convert('RGB')
#         image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
#         image.save(jpath(dataroot, impath))
#     print()

import argparse
import numpy as np
from os.path import join as jpath
from PIL import Image
import os

def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='datasets/KingsCollege', help='dataset root')
    parser.add_argument('--height', type=int, default=256, help='image height')
    parser.add_argument('--width', type=int, default=455, help='image width')
    parser.add_argument('--save_resized_imgs', action="store_true", default=False, help='save resized train/test images [height, width]')
    return parser.parse_args()

args = params()
dataroot = args.dataroot
imsize = [args.height, args.width]  # (H, W)

try:
    with open(jpath(dataroot, 'dataset_train.txt'), 'r', encoding='utf-8') as f:
        imlist = []
        
        for line in f:
            parts = line.strip().split()
            if parts[0] in ('a', 'b'):  # if the first part is 'a' or 'b'     
                imlist.append(f"{parts[0]} {parts[1]}")
            else:
                imlist.append(parts[0])
    print(f"Loaded image list from {jpath(dataroot, 'dataset_train.txt')}")
except Exception as e:
    print(f"Error loading image list: {e}")
    exit(1)

mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=np.float64)

for i, impath in enumerate(imlist):
    try:
        full_path = jpath(dataroot, impath)
        print(f'[{i+1}/{len(imlist)}]: Processing {full_path}', end='\r')
        image = Image.open(full_path).convert('RGB')
        image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
        mean_image += np.array(image).astype(np.float64)

        # save resized training images
        if args.save_resized_imgs:
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))
            image.save(full_path)
    except Exception as e:
        print(f"Error processing {full_path}: {e}")

print()
mean_image /= len(imlist)
Image.fromarray(mean_image.astype(np.uint8)).save(jpath(dataroot, 'mean_image.png'))
np.save(jpath(dataroot, 'mean_image.npy'), mean_image)

# save resized test images
if args.save_resized_imgs:
    try:
        with open(jpath(dataroot, 'dataset_test.txt'), 'r', encoding='utf-8') as f:
            imlist = []
            for line in f:
                parts = line.strip().split()
                if parts[0] in ('a', 'b'):  # if the first part is 'a' or 'b'
                    imlist.append(f"{parts[0]} {parts[1]}")
                else:
                    imlist.append(parts[0])
        for i, impath in enumerate(imlist):
            try:
                full_path = jpath(dataroot, impath)
                print(f'[{i+1}/{len(imlist)}]: Processing {full_path}', end='\r')
                image = Image.open(full_path).convert('RGB')
                image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
                image.save(full_path)
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
        print()
    except Exception as e:
        print(f"Error loading test image list: {e}")
