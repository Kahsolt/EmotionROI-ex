#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.segmentation.fcn import fcn_resnet50, FCN_ResNet50_Weights, FCN
import matplotlib.pyplot as plt

from data import EmotionROI, DataLoader
from utils import *

# according to the essay, and also follow: https://arxiv.org/abs/1411.4038
# NOTE: this does NOT work!!
if not 'follow essay':
  LR = 1e-8
  EPOCHS = 20
  BATCH_SIZE = 20
else:
  LR = 1e-2
  EPOCHS = 100
  BATCH_SIZE = 20


def F_beta(prec:float, recall:float, beta:float) -> float:
  ''' 原论文计算F值之前需要对图进行二值化，依据为1979年的论文 https://engineering.purdue.edu/kak/computervision/ECE661.08/OTSU_paper.pdf 这实在是太复杂并且不合理了 '''
  return (1 + beta**2) * (prec * recall) / (beta**2 * prec + recall)


def get_model() -> FCN:
  from torch.nn import Conv2d

  # 预训练的FCN模型；但原论文使用 PASCAL VOC 上预训练的 AlexNet 为骨干
  model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
  # 最后一层，解释为单层灰度图 并 全零初始化
  layer: Conv2d = model.classifier[-1]
  kwargs = {
    'in_channels': layer.in_channels,
    'kernel_size': layer.kernel_size,
    'stride': layer.stride,
    'padding': layer.padding,
    'dilation': layer.dilation,
    'groups': layer.groups,
    'bias': layer.bias is not None,
    'padding_mode': layer.padding_mode,
  }
  new_layer = Conv2d(out_channels=1, **kwargs)
  init.zeros_(new_layer.weight)
  if new_layer.bias is not None:
    init.zeros_(new_layer.bias)
  model.classifier[-1] = new_layer

  return model


def train(args):
  seed_everything()

  train_dataset = EmotionROI('train')
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
  test_dataset = EmotionROI('test')
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

  model = get_model().to(device)
  optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
  scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7, verbose=True)

  loss_train, loss_test = [], []
  step = 0
  for epoch in range(EPOCHS):
    ''' Train '''
    model.train()
    tot, mse = 0, 0.0
    for X, Y in tqdm(train_loader):
      X, Y = X.to(device), Y.to(device)

      optimizer.zero_grad()
      output = model(X)['out']
      loss_raw = F.mse_loss(output, Y, reduction='none')
      loss = loss_raw.mean()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        mse += loss_raw.mean(dim=[1, 2, 3]).sum().item()
        tot += len(Y)

      step += 1
      if step % 10 == 0:
        print(f'>> [step {step}] loss: {loss.item()}')

    loss_train.append(mse / tot)
    scheduler.step()

    ''' Eval '''
    tot, mse = 0, 0.0
    with torch.inference_mode():
      model.eval()
      for X, Y in tqdm(test_loader):
        X, Y = X.to(device), Y.to(device)

        output = model(X)['out']
        loss_raw = F.mse_loss(output, Y, reduction='none')
        mse += loss_raw.mean(dim=[1, 2, 3]).sum().item()
        tot += len(Y)

      print(f'>> [Epoch: {epoch + 1}/{EPOCHS}] mse: {mse / tot:.7f}')

    loss_test.append(mse / tot)

  if 'plot':
    plt.clf()
    plt.plot(loss_train, 'r', label='train')
    plt.plot(loss_test, 'b', label='test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_PATH / 'FCNEL-loss.png', dpi=600)

  torch.save(model.state_dict(), LOG_PATH / 'model-FCNEL.pth')


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.parse_args()

  train(args)
