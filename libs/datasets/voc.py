from __future__ import print_function

import os
import os.path as osp
import sys

import torch
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np


class VOCSegmentation(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.mean_rgb = np.array([123.68, 116.779, 103.939])

        self._annopath = osp.join(self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = osp.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = osp.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id)
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = (np.array(img) - self.mean_rgb).transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))
        target = np.array(target, dtype=np.int32)
        target[target == 255] = -1
        target = torch.from_numpy(target.astype(np.int64))

        return img, target

    def __len__(self):
        return len(self.ids)
