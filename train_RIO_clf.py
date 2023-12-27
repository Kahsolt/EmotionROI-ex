#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.resnet import resnet50, ResNet50_Weights, ResNet
import matplotlib.pyplot as plt

from data import Emotion6Hard, DataLoader
from utils import *

# Can we directly do classification on only ROI annotations?

LR = 1e-1
EPOCHS = 130
BATCH_SIZE = 16


def get_model() -> ResNet:
  from torch.nn import Conv2d, Linear

  # 预训练的ResNet模型
  model = resnet50(weights=ResNet50_Weights.DEFAULT)
  # 修改第一层输入通道数
  layer = model.conv1
  kwargs = {
    'out_channels': layer.out_channels,
    'kernel_size': layer.kernel_size,
    'stride': layer.stride,
    'padding': layer.padding,
    'dilation': layer.dilation,
    'groups': layer.groups,
    'bias': layer.bias is not None,
    'padding_mode': layer.padding_mode,
  }
  new_layer = Conv2d(in_channels=1, **kwargs)
  new_layer.weight.data = layer.weight.data.mean(dim=1, keepdims=True)
  model.conv1 = new_layer
  # 修改最后一层输出单元数
  layer = model.fc
  kwargs = {
    'in_features': layer.in_features,
    'out_features': Emotion6Hard.n_class,
    'bias': layer.bias is not None,
  }
  new_layer = Linear(**kwargs)
  if new_layer.bias is not None:
    init.zeros_(new_layer.bias)
  model.fc = new_layer

  return model


def train(args):
  seed_everything()

  train_dataset = Emotion6Hard('train')
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
  test_dataset = Emotion6Hard('test')
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

  model = get_model().to(device)
  optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
  scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7, verbose=True)

  acc_train, acc_test = [], []
  step = 0
  for epoch in range(EPOCHS):
    ''' Train '''
    model.train()
    tot, ok = 0, 0
    for X, Y in tqdm(train_loader):
      X, Y = X.to(device), Y.to(device)

      optimizer.zero_grad()
      output = model(X)
      loss = F.cross_entropy(output, Y)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        ok += (torch.argmax(output, dim=-1) == Y).sum().item()
        tot += len(Y)

      step += 1
      if step % 10 == 0:
        print(f'>> [step {step}] loss: {loss.item()}, acc: {ok / tot:.3%}')

    acc_train.append(ok / tot)
    scheduler.step()

    ''' Eval '''
    tot, ok = 0, 0
    with torch.inference_mode():
      model.eval()
      for X, Y in tqdm(test_loader):
        X, Y = X.to(device), Y.to(device)

        output = model(X)
        ok += (torch.argmax(output, dim=-1) == Y).sum().item()
        tot += len(Y)

      print(f'>> [Epoch: {epoch + 1}/{EPOCHS}] acc: {ok / tot:.3%}')

    acc_test.append(ok / tot)

  if 'plot':
    plt.clf()
    plt.plot(acc_train, 'r', label='train')
    plt.plot(acc_test, 'b', label='test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_PATH / 'RIO_clf-acc.png', dpi=600)

  torch.save(model.state_dict(), LOG_PATH / 'model-RIO_clf.pth')


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.parse_args()

  train(args)
