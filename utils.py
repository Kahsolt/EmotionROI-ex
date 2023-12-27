#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/11 

import random
from PIL import Image
from PIL.Image import Resampling
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import *

import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
IMG_PATH = BASE_PATH / 'img'

npimg_u8 = NDArray[np.uint8]
npimg_f32 = NDArray[np.float32]
npimg = Union[npimg_u8, npimg_f32]


def seed_everything(seed:int=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def param_cnt(model:Module) -> int:
  return sum([p.numel() for p in model.parameters()])


def read_txt(fp:Path) -> List[str]:
  with open(fp, 'r', encoding='utf-8') as fh:
    return fh.read().strip().split('\n')


def load_pil(fp:Path, mode:str=None) -> PILImage:
  return Image.open(fp).convert(mode)

def load_im(fp:Path, mode:str=None, dtype=np.float32) -> npimg:
  img = load_pil(fp, mode)
  im = pil_to_npimg(img, dtype)
  return im

def pil_to_npimg(img:PILImage, dtype=np.float32) -> npimg:
  assert dtype in [np.uint8, np.float32, 'uint8', 'float32']
  im = np.asarray(img, dtype=np.uint8)
  if dtype in [np.float32, 'float32']:
    im = im.astype(np.float32) / 255.0
  return im

def npimg_to_pil(im:npimg) -> PILImage:
  assert isinstance(im, ndarray)
  assert len(im.shape) == 3 and im.shape[-1] == 3
  if im.dtype == np.float32:
    assert 0.0 <= im.min() and im.max() <= 1.0
    im = np.asarray(im * 255, dtype=np.uint8)
  else:
    assert im.dtype == np.uint8
  return Image.fromarray(im)

def im_f32_t2_u8(im:npimg) -> npimg:
  if im.dtype == np.uint8: return im
  if not (0.0 <= im.min() and im.max() <= 1.0):
    print(f'>> clip vrng [{im.min()}, {im.max()}]')
    im = np.clip(im, 0.0, 1.0)
  return (im * 255).astype(np.uint8)
