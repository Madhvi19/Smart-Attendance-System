from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os.path as osp
import glob
from skimage.transform import rescale, resize
from skimage.io import imsave, imread

class FaceLandmarksDataset():
    def __init__(self, root_dir, model_type="recognition", transform=None):
        self.data_list = glob.glob(osp.join(root_dir, '*.jpg'))
        labels = []
        for d in self.data_list:
            data = d.split("/")
            data = data[2].split("_")
            labels.append(data[0])
        self.labels = labels


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data_list[idx]
        
        # Image
        img = imread(image)
        img1 = resize(img, (224,224,3))
        image = torch.from_numpy(img1)
#         image = imread(img1)
#         print("sahpe",image2.shape)
        # One Hot Labels
        label = self.labels[idx]

        gt_label = np.zeros((11),np.float32)
        for i in range(11):
            if int(label) == i:
                gt_label[i] = 1.0

        
        # Sample
        sample = {'image': image, 'gt_label': gt_label}
        print(sample)
        return sample