# import os.path
# import torchvision.transforms as transforms
# from data.base_dataset import BaseDataset, get_posenet_transform
# from data.image_folder import make_dataset
# from PIL import Image
# import PIL
# import random
# import numpy

# class UnalignedPoseNetDataset(BaseDataset):
#     def initialize(self, opt):
#         self.opt = opt
#         self.root = opt.dataroot

#         split_file = os.path.join(self.root , 'dataset_'+opt.phase+'.txt')
#         self.A_paths = numpy.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
#         self.A_paths = [os.path.join(self.root, path) for path in self.A_paths]
#         self.A_poses = numpy.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1,2,3,4,5,6,7))
#         self.mean_image = numpy.load(os.path.join(self.root , 'mean_image.npy'))
#         if opt.model == "poselstm":
#             self.mean_image = None
#             print("mean image subtraction is deactivated")

#         self.A_size = len(self.A_paths)
#         self.transform = get_posenet_transform(opt, self.mean_image)

#     def __getitem__(self, index):
#         A_path = self.A_paths[index % self.A_size]
#         index_A = index % self.A_size
#         # print('(A, B) = (%d, %d)' % (index_A, index_B))
#         A_img = Image.open(A_path).convert('RGB')
#         A_pose = self.A_poses[index % self.A_size]

#         A = self.transform(A_img)

#         return {'A': A, 'B': A_pose,
#                 'A_paths': A_path}

#     def __len__(self):
#         return self.A_size

#     def name(self):
#         return 'UnalignedPoseNetDataset'



# import os
# import pandas as pd
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from PIL import Image
# import torch
# import numpy as np
# #ours
# class UnalignedPoseNetDataset(Dataset):
#     def __init__(self, csv_file, image_folder, mean_image_path=None, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.image_folder = image_folder
#         self.transform = transform
#         if mean_image_path:
#             self.mean_image = torch.tensor(np.load(mean_image_path)).float()
#         else:
#             self.mean_image = None

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_folder, self.data_frame.iloc[idx, 0])
#         image = Image.open(img_name).convert('RGB')
#         pose = self.data_frame.iloc[idx, 1:].values.astype('float32')
#         if self.transform:
#             image = self.transform(image)
#         return {'A': image, 'B': torch.tensor(pose), 'A_paths': img_name}

# def get_posenet_transform(mean_image=None):
#     transform_list = [transforms.ToTensor()]
#     if mean_image is not None:
#         transform_list.append(transforms.Normalize(mean=mean_image, std=[1.0, 1.0, 1.0]))
#     return transforms.Compose(transform_list)

import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

def get_default_transform(mean_image_path):
    transform_list = [transforms.ToTensor()]
    if mean_image_path:
        mean_image = np.load(mean_image_path)
        mean_image = mean_image.transpose(2, 0, 1)  # [H, W, C] to [C, H, W]
        mean = mean_image.mean(axis=(1, 2)) / 255.0
        std = mean_image.std(axis=(1, 2)) / 255.0
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)

class UnalignedPoseNetDataset(Dataset):
    def __init__(self, csv_file, image_folder, mean_image_path=None, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform if transform else get_default_transform(mean_image_path)
        self.mean_image = torch.tensor(np.load(mean_image_path)).float() if mean_image_path else None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]  # 이미지 이름
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # 위치 및 방향 데이터
        pose = self.data_frame.iloc[idx, 1:7].values.astype('float32')

        if self.transform:
            image = self.transform(image)

        return {'A': image, 'B': torch.tensor(pose), 'A_paths': img_path}
