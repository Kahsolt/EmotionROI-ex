#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils import *
from utils import Path

DATA_PATH = BASE_PATH / 'data'
DATA_EMOTIONROI_PATH = DATA_PATH / 'EmotionROI'

RESIZE = (224, 224)

# torchvision pretrained using ImageNet stats 
TV_MEAN = [0.485, 0.456, 0.406]
TV_STD  = [0.229, 0.224, 0.225]


class ComposeDual(T.Compose):

  def __call__(self, img, ant) -> tuple:
    for t in self.transforms:
      img, ant = t(img, ant)
    return img, ant

class RandomResizedCropDual(T.RandomResizedCrop):

  def forward(self, img, ant) -> tuple:
    i, j, h, w = self.get_params(img, self.scale, self.ratio)
    img = TF.resized_crop(img, i, j, h, w, self.size, T.InterpolationMode.BILINEAR,      antialias=True)
    ant = TF.resized_crop(ant, i, j, h, w, self.size, T.InterpolationMode.NEAREST_EXACT, antialias=False)
    return img, ant

class RandomHorizontalFlipDual(T.RandomHorizontalFlip):

  def forward(self, img, ant) -> tuple:
    if torch.rand(1) < self.p:
      img = TF.hflip(img)
      ant = TF.hflip(ant)
    return img, ant


class EmotionROI(Dataset):

  class_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
  n_class = len(class_names)

  transform_image = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=TV_MEAN, std=TV_STD),
  ])
  transform_annot = T.Compose([
    T.ToTensor(),
  ])
  transform_aug = ComposeDual([
    RandomResizedCropDual(RESIZE),
    RandomHorizontalFlipDual(),
  ])

  def __init__(self, split:str='train', ret_label:bool=False, root:Path=DATA_EMOTIONROI_PATH):
    assert split in ['train', 'test'], f'>> split should be "train" or "test", but got: {split}'

    self.root = root
    self.split = split
    self.ret_label = ret_label
    self.img_dp = root / 'images'
    self.ant_dp = root / 'ground_truth'
    lst_dp = root / 'training_testing_split'
    self.metadata = read_txt(lst_dp / f'{split}ing.txt')

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx:int):
    fn = self.metadata[idx]
    image = self.transform_image(load_pil(self.img_dp / fn))
    annot = self.transform_annot(load_pil(self.ant_dp / fn))
    if self.split == 'train':
      image, annot = self.transform_aug(image, annot)
    if self.ret_label:
      label = self.class_names.index(fn.split('/')[0])
      return image, annot, label
    else:
      return image, annot


class Emotion6Hard(EmotionROI):

  transform_aug = T.Compose([
    T.RandomResizedCrop(RESIZE, interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=False),
    T.RandomHorizontalFlip(),
  ])

  def __getitem__(self, idx:int):
    fn = self.metadata[idx]
    annot = self.transform_annot(load_pil(self.ant_dp / fn))
    if self.split == 'train':
      annot = self.transform_aug(annot)
    label = self.class_names.index(fn.split('/')[0])
    return annot, label


if __name__ == '__main__':
  dataset = EmotionROI()
  for X, Y in iter(dataset):
    print(X.shape)
    print(Y.shape)
    break

  dataset = Emotion6Hard()
  for X, Y in iter(dataset):
    print(X.shape)
    print(Y)
    break
