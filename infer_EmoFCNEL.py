#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

from torchvision.utils import make_grid

from infer_FCNEL import *
from infer_FCNEL import App as AppBase
from train_EmoFCNEL import get_model


@torch.inference_mode()
def infer(model:FCN, X:Tensor) -> npimg_u8:
  X = X.unsqueeze_(0).expand(EmotionROI.n_class, -1, -1, -1).to(device)
  E = torch.LongTensor(range(EmotionROI.n_class)).to(X.device)
  output = model(X, E)['out']
  pred = torch.clamp(output, 0.0, 1.0)
  pred = make_grid(pred, nrow=2).permute([1, 2, 0]).cpu().numpy()
  return im_f32_t2_u8(pred)


class App(AppBase):

  def load_model(self):
    args = self.args
    model: FCN = get_model()
    fp = args.load or (LOG_PATH / f'model-{args.model}.pth')
    print(f'>> load weights from {fp}')
    state_dict = torch.load(fp, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    self.model = model

  def load_dataset(self, split:str):
    if split == 'train' and self.trainset is None:
      self.trainset = EmotionROI('train', ret_label=True)
    if split == 'test' and self.testset is None:
      self.testset = EmotionROI('test', ret_label=True)

  def run_infer(self, X:Tensor) -> npimg:
    return infer(self.model, X)


if __name__ == '__main__':
  args = get_args()
  args.model = 'EmoFCNEL'
  App(args)
