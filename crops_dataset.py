from __future__ import print_function
from glob import glob
import torch.utils.data as data
import os
import cv2
from PIL import Image
from utils import preprocess
import random
import numpy as np


class CropSegmentation(data.Dataset):
  CLASSES = ['background', 'tobacco', 'cron', 'barley-rice']

  def __init__(self, train=True, transform=None, target_transform=None, download=False, crop_size=None):
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size

    if download:
      self.download()
    imgpaths = sorted(glob('../data/train/data' + '/*.jpg'))
    maskpaths = sorted(glob('../data/train/label' + '/*.png'))
    
    print(len(imgpaths),len(maskpaths))
    
    s1=[]
    s2=[]
    s3=[]
    im1=[]
    im2=[]
    im3=[]
    print("load dataset....")
    for i in range(len(maskpaths)):
        img=cv2.imread(maskpaths[i])
        if img.max()==3:
            s3.append(maskpaths[i])
            im3.append(imgpaths[i])
        if img.max()==2:
            s2.append(maskpaths[i])
            im2.append(imgpaths[i])
        if img.max()==1:
            s1.append(maskpaths[i])
            im1.append(imgpaths[i])
        print(i+1,3905)
    
    random_train = list(range(len(im1)))
    random.seed(105)
    random.shuffle(random_train)
    im11 = [im1[idx] for idx in random_train]
    s11 =[s1[idx] for idx in random_train]
    
    random_train = list(range(len(im2)))
    random.seed(105)
    random.shuffle(random_train)
    im22 =[im2[idx] for idx in random_train]
    s22 = [s2[idx] for idx in random_train]
    
    random_train = list(range(len(im3)))
    random.seed(105)
    random.shuffle(random_train)
    im33 = [im3[idx] for idx in random_train]
    s33 = [s3[idx] for idx in random_train]
    
    
    
    train_image = im11[:670]+im22[:640]+im33[:2200]
    train_mask = s11[:670]+s22[:640]+s33[:2200]
    test_image = im11[670:]+im22[640:]+im33[2200:]
    test_mask = s11[670:]+s22[640:]+s33[2200:]
    
    print(len(train_image),len(train_mask),len(test_image),len(test_mask))
    
    
#     random_train = list(range(len(train_image)))
#     random.seed(105)
#     random.shuffle(random_train)
#     #print(random_train)
#     train_img_paths1 = np.array([train_image[idx] for idx in random_train])
#     train_scores1 = np.array([train_mask[idx] for idx in random_train])
    
    self.images = []
    self.masks = []
    if self.train:
        self.images = train_image
        self.masks = train_mask 
      
    else:
        self.images = test_image
        self.masks = test_mask
      

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])
#     _target
#     _img, _target = preprocess2(_img, _target)
    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size))


    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = self.target_transform(_target)

    return _img, _target

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')

