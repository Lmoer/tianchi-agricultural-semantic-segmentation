import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

def crop_inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
#   pred += 1
#   mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)


def preprocess(image, mask, flip=False, scale=None, crop=None):
  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = image.resize(new_size, Image.ANTIALIAS)
    mask = mask.resize(new_size, Image.NEAREST)

  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  image = data_transforms(image)
  mask = torch.LongTensor(np.array(mask).astype(np.int64))

  if crop:
    h, w = image.shape[1], image.shape[2]
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)
    image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
    mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 0)(mask)
#     mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

    h, w = image.shape[1], image.shape[2]
    i = random.randint(0, h - crop[0])
    j = random.randint(0, w - crop[1])
    image = image[:, i:i + crop[0], j:j + crop[1]]
    mask = mask[i:i + crop[0], j:j + crop[1]]

  return image, mask


def preprocess1(image, mask, flip=True, scale=True):
    data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        if w > h:
            new_size = (512, int(512 / w * h))
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image = data_transforms(image)
            mask = torch.LongTensor(np.array(mask).astype(np.int64))
            image = torch.nn.ZeroPad2d((0, 0, 0, 512 - new_size[1]))(image)
            mask = torch.nn.ConstantPad2d((0, 0, 0, 512 - new_size[1]), 0)(mask)
        elif w < h:
            new_size = (int(512 / h * w), 512)
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image = data_transforms(image)
            mask = torch.LongTensor(np.array(mask).astype(np.int64))
            image = torch.nn.ZeroPad2d((0, 512 - new_size[0], 0, 0))(image)
            mask = torch.nn.ConstantPad2d((0, 512 - new_size[0], 0, 0), 0)(mask)
        else:
            new_size = (512, 512)
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image = data_transforms(image)
            mask = torch.LongTensor(np.array(mask).astype(np.int64))
        
        
        

#   data_transforms = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#   image = data_transforms(image)
#   mask = torch.LongTensor(np.array(mask).astype(np.int64))


    return image, mask


def preprocess2(image, mask, flip=True, scale=True):
    data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        if w > h:
            new_size = (512, int(512 / w * h))
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image1 = np.zeros((512,512,3))
            image1[0:new_size[1],:,:] = np.array(image)
            image1 = data_transforms(image1.astype(np.uint8))
            mask = torch.LongTensor(np.array(mask).astype(np.int64))
            mask = torch.nn.ConstantPad2d((0, 0, 0, 512 - new_size[1]), 0)(mask)
        elif w < h:
            new_size = (int(512 / h * w), 512)
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image1 = np.zeros((512,512,3))
            image1[:,:new_size[0],:] = np.array(image)
            image1 = data_transforms(image1.astype(np.uint8))
            mask = torch.LongTensor(np.array(mask).astype(np.int64))
            mask = torch.nn.ConstantPad2d((0, 512 - new_size[0], 0, 0), 0)(mask)
        else:
            new_size = (512, 512)
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
            image1 = data_transforms(image)
            mask = torch.LongTensor(np.array(mask).astype(np.int64))


    return image1, mask
