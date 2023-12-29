#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/29 

import seaborn as sns
import matplotlib.pyplot as plt

from infer_EmoFCNEL import *

LABELS = EmotionROI.class_names


def calc_dist_pairwise(x:Tensor, y:Tensor, metric:str) -> Tensor:
  if metric == 'L1':
    dist = (x.unsqueeze(dim=1) - y.unsqueeze(dim=0)).abs().mean(dim=-1)           # range in [0, inf]
  elif metric == 'L2':
    dist = (x.unsqueeze(dim=1) - y.unsqueeze(dim=0)).square().sum(dim=-1).sqrt()  # range in [0, inf]
  elif metric in ['Cosine', 'Cosine Abs']:
    x_n = x.norm(dim=-1, keepdim=True)
    y_n = y.norm(dim=-1, keepdim=True)
    cosim = (x @ y.T) / ((x_n @ y_n.T) + 1e-8)    # range in [-1, 1]
    if metric == 'Cosine Abs':
      cosim = cosim.abs()                         # range in [0, 1]
    dist = 1 - cosim                              # range in [0, 2] or [0, 1]
  else: raise ValueError(f'unknown dist_metric {metric}')
  return dist


def vis(args):
  fp = args.load or (LOG_PATH / f'model-{args.model}.pth')
  state_dict = torch.load(fp, map_location='cpu')
  embed: Tensor = state_dict['embd.weight']
  print('embed.shape [N, D]:', tuple(embed.shape))

  dist_L1     = calc_dist_pairwise(embed, embed, 'L1')
  dist_L2     = calc_dist_pairwise(embed, embed, 'L2')
  dist_Cos    = calc_dist_pairwise(embed, embed, 'Cosine')
  dist_CosAbs = calc_dist_pairwise(embed, embed, 'Cosine Abs')
  
  kwargs = {
    'cbar': True, 
    'cmap': 'binary',
    'xticklabels': LABELS, 
    'yticklabels': LABELS,
  }
  plt.clf()
  plt.subplot(221) ; plt.title('L1')     ; sns.heatmap(dist_L1,     **kwargs)
  plt.subplot(222) ; plt.title('L2')     ; sns.heatmap(dist_L2,     **kwargs)
  plt.subplot(223) ; plt.title('Cos')    ; sns.heatmap(dist_Cos,    **kwargs, vmin=0, vmax=2)
  plt.subplot(224) ; plt.title('CosAbs') ; sns.heatmap(dist_CosAbs, **kwargs, vmin=0, vmax=1)
  plt.suptitle('embed dist')
  plt.tight_layout()
  plt.savefig(IMG_PATH / 'EmoFCNEL-embed.png', dpi=600)
  plt.show()


if __name__ == '__main__':
  args = get_args()
  args.model = 'EmoFCNEL'
  vis(args)
