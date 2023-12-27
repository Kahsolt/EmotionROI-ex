#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from argparse import ArgumentParser, Namespace
from traceback import print_exc

import torch
from torchvision.models.segmentation.fcn import FCN
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import EmotionROI
from utils import *

from train_FCNEL import get_model as get_model_FCNEL

device = 'cpu'

WINDOW_TITLE = 'ROI interactive'
WINDOW_SIZE  = (800, 500)
FIG_SIZE     = (30, 20)

DATA_SOURCES = ['train', 'test', '<file>']
DEFAULT_DATA_SOURCE = 'test'
TITLES = ['input', 'pred', 'truth']
CMAPS  = [None, 'grey', 'grey']


@torch.inference_mode()
def infer(model:FCN, X:Tensor) -> npimg_u8:
  X = X.unsqueeze_(0).to(device)
  output = model(X)['out']
  pred = torch.clamp(output[0], 0.0, 1.0).squeeze().cpu().numpy()
  return im_f32_t2_u8(pred)


class App:

  def __init__(self, args:Namespace):
    self.args: Namespace = args
    self.model: FCN = None
    self.trainset: EmotionROI = None
    self.testset: EmotionROI = None

    self.data_src_memo = {
      'train': 0,
      'test': 0,
      '<file>': '',
    }

    self.setup_gui()
    self.init_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def init_workspace(self):
    seed_everything()

    model: FCN = globals()[f'get_model_{args.model}']()
    fp = args.load or (LOG_PATH / f'model-{args.model}.pth')
    print(f'>> load weights from {fp}')
    state_dict = torch.load(fp, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    self.model = model
    self._change_data_source()
    self._change_dataset_idx()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    #wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: control
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_data_src = tk.StringVar(frm1, value=DEFAULT_DATA_SOURCE)
      cb = ttk.Combobox(frm1, state='readonly', values=DATA_SOURCES, textvariable=self.var_data_src)
      cb.bind('<<ComboboxSelected>>', lambda evt: self._change_data_source())
      cb.pack(side=tk.LEFT, expand=tk.NO)

      self.var_fp = tk.StringVar(wnd, value='')
      tk.Entry(frm1, textvariable=self.var_fp).pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
      tk.Button(frm1, text='Open..', command=self._open_file).pack(side=tk.RIGHT)

    # top: control
    frm11 = ttk.Frame(wnd)
    frm11.pack(expand=tk.YES, fill=tk.X)
    if True:
      self.var_idx = tk.IntVar(wnd, value=0)
      sc = tk.Scale(frm11, variable=self.var_idx, 
                    from_=0, to=10, resolution=1, tickinterval=100, orient=tk.HORIZONTAL, 
                    command=lambda evt: self._change_dataset_idx())
      sc.pack(expand=tk.YES, fill=tk.X)
      self.sc = sc

    # mid: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      fig = plt.figure(figsize=FIG_SIZE)
      fig.clear()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.draw()
      cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
      self.fig: Figure = fig
      self.cvs: FigureCanvasTkAgg = cvs

  def _change_data_source(self):
    src = self.var_data_src.get()
    if src == 'train':
      if self.trainset is None:
        self.trainset = EmotionROI('train')
      self.sc.config(to=len(self.trainset) - 1)
      self.var_idx.set(self.data_src_memo[src])
    elif src == 'test':
      if self.testset is None:
        self.testset = EmotionROI('test')
      self.sc.config(to=len(self.testset) - 1)
      self.var_idx.set(self.data_src_memo[src])
    elif src == '<file>':
      self.var_fp.set(self.data_src_memo[src])
    else: raise ValueError(src)

  def _change_dataset_idx(self):
    src = self.var_data_src.get()
    if   src == 'train': dataset = self.trainset
    elif src == 'test':  dataset = self.testset
    else: return

    idx = self.var_idx.get()
    fn = dataset.metadata[idx]
    X = dataset.transform_annot(load_pil(dataset.img_dp / fn))
    Y = dataset.transform_annot(load_pil(dataset.ant_dp / fn))

    tc_to_u8 = lambda x: im_f32_t2_u8(x.permute([1, 2, 0]).squeeze().cpu().numpy())
    ims = [
      tc_to_u8(X),
      infer(self.model, X),
      tc_to_u8(Y),
    ]
    self.data_src_memo[src] = idx
    self.plot(ims)

  def _open_file(self):
    fp = tkfdlg.askopenfilename()
    if not fp: return
    if not Path(fp).is_file():
      tkmsg.showerror('Error', f'path {fp} is not a file!')
      return

    self.var_data_src.set('<file>')
    self._change_data_source()

    self.var_fp.set(fp)
    print(f'[image] load image from {fp!s}')

    # images
    ims = []
    # fig 1: original
    fp = Path(fp)
    img = load_pil(fp, mode='RGB')
    ims.append(pil_to_npimg(img, dtype=np.uint8))
    # fig 2: prediction
    X: Tensor = EmotionROI.transform_image(img)
    ims.append(infer(self.model, X))
    # fig 3: ground truth (optional)
    fp_gt = Path(str(fp).replace('\\', '/').replace('/images/', '/ground_truth/'))
    has_gt = fp_gt.is_file()
    if has_gt: ims.append(load_im(fp_gt, mode='L', dtype=np.uint8))
    # plot
    self.data_src_memo['<file>'] = fp
    self.plot(ims)

  def plot(self, ims:List[npimg_u8]):
    self.fig.clear()
    axs = self.fig.subplots(1, len(ims))
    for i, ax in enumerate(axs):
      ax.imshow(ims[i], cmap=CMAPS[i])
      ax.set_title(TITLES[i])
      ax.set_axis_off()
    self.cvs.draw()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='FCNEL', choices=['FCNEL', 'EmoFCNEL'])
  parser.add_argument('--load', type=Path, help='pretrained ckpt path')
  args = parser.parse_args()

  App(args)
