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
    T.Normalize(mean=[0.4165, 0.3834, 0.3488], std=[0.2936, 0.2805, 0.2850]),
  ])
  transform_annot = T.Compose([
    T.ToTensor(),
  ])
  transform_aug = ComposeDual([
    RandomResizedCropDual(RESIZE),
    RandomHorizontalFlipDual(),
  ])

  def __init__(self, split:str='train', root:Path=DATA_EMOTIONROI_PATH):
    assert split in ['train', 'test'], f'>> split should be "train" or "test", but got: {split}'

    self.root = root
    self.split = split
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
    return image, annot

  def show_stats(self):
    ims = []
    for dp in tqdm((self.img_dp.iterdir())):
      for fp in tqdm(list(dp.iterdir())):
        img = load_pil(fp)
        img = img.resize(RESIZE, resample=Resampling.BILINEAR)
        ims.append(pil_to_npimg(img))
    X = torch.from_numpy(np.stack(ims, axis=0))
    X = X.permute(0, 3, 1, 2)

    # [N=1980, C=3, H=224, W=224]
    print('X.shape', X.shape)
    # [0.4165, 0.3834, 0.3488]
    print('mean:', X.mean(axis=[0, 2, 3]))
    # [0.2936, 0.2805, 0.2850]
    print('std:',  X.std (axis=[0, 2, 3]))


if __name__ == '__main__':
  dataset = EmotionROI()
  #dataset.show_stats()

  for X, Y in iter(dataset):
    print(X.shape)
    print(Y.shape)
    break
