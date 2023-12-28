#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/28

import torch.nn as nn
from torchvision.models.resnet import ResNet

from train_FCNEL import *

LR = 1e-2
EPOCHS = 100
BATCH_SIZE = 20


class EmoFCNEL(nn.Module):

  def __init__(self, d_embd:int=6):
    super().__init__()

    # 基础FCN模型
    self.base = get_hijacked_model()
    # 标签嵌入
    self.embd = nn.Embedding(num_embeddings=EmotionROI.n_class, embedding_dim=d_embd)
  
  def forward(self, x:Tensor, e:Tensor) -> Dict[str, Tensor]:
    B, C, H, W = x.shape
    emo: Tensor = self.embd(e).unsqueeze(-1).unsqueeze(-1)
    emo_expand = emo.expand(-1, -1, H, W)
    fused = torch.cat([x, emo_expand], dim=1)
    return self.base(fused)


def get_hijacked_model(d_embd:int=6) -> FCN:
  from torch.nn import Conv2d
  from train_FCNEL import get_model as get_model_FCNEL

  # 基础FCNEL模型
  model = get_model_FCNEL()
  # 第一层，扩充输入深度
  backbone: ResNet = model.backbone
  layer: Conv2d = backbone.conv1
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
  new_layer = Conv2d(in_channels=3+d_embd, **kwargs)
  new_layer.weight.data[:, :3, :, :] = layer.weight.data
  backbone.conv1 = new_layer

  return model


def get_model(d_embd:int=6) -> EmoFCNEL:
  return EmoFCNEL(d_embd)


def train(args):
  seed_everything()

  train_dataset = EmotionROI('train', ret_label=True)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
  test_dataset = EmotionROI('test', ret_label=True)
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
    for X, Y, E in tqdm(train_loader):
      X, Y, E = X.to(device), Y.to(device), E.to(device)

      optimizer.zero_grad()
      output = model(X, E)['out']
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
      for X, Y, E in tqdm(test_loader):
        X, Y, E = X.to(device), Y.to(device), E.to(device)

        output = model(X, E)['out']
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
    plt.savefig(IMG_PATH / f'{args.model}-loss.png', dpi=600)

  torch.save(model.state_dict(), LOG_PATH / f'model-{args.model}.pth')


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.parse_args()

  args.model = 'EmoFCNEL'

  train(args)
