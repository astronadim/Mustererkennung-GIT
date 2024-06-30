import random
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def load_img(row, col):
  fname = 'top_potsdam_%d_%d_RGB.tif' % (row, col)
  img = Image.open('../data/2_Ortho_RGB/' +fname)
  img = np.array(img)
  return img


def load_crops(row, col):
  fname = 'objects_potsdam_%d_%d_crops.mat' % (row, col)
  data = scipy.io.loadmat('../data/7_Objects_Crops_cs128_p10/' + fname)
  images = data['crops']
  labels = data['labels']
  return images, labels


def load_bbs(row, col):
  fname = 'objects_potsdam_%d_%d_boundingBoxes.mat' % (row, col)
  data = scipy.io.loadmat('../data/6_Objects_BBs/' + fname)
  oc = data['object_classes'][0]
  # colors = [c[0][0][3][0] for c in oc]
  bbs = [c[0][0][4] for c in oc]
  bbs = np.concatenate(bbs)
  return bbs


def plot_samples(crops, gt, class_names, n=8):
  fig = plt.figure(figsize=(15, 30))
  grid = ImageGrid(fig, 111, nrows_ncols=(1, n), axes_pad=0.01)
  for i, ax in enumerate(grid):
      i = random.randrange(0, crops.shape[-1])
      ax.imshow(crops[:,:,0:3,i])
      ax.set_title(class_names[gt[i].item()])
  plt.show()


def plot_bbs(img, bbs, class_labels, bb_labels=None, colors=None, title=''):
  fig, ax = plt.subplots()
  plt.imshow(img)
  ax.set_title(title)
  for i in range(bbs.shape[0]):
    bb = bbs[i,:]
    if colors is None:
      c = 'k'
    else:
      c = colors[class_labels[i]]
    rect = Rectangle(bb[0:2], bb[2], bb[3], linewidth=0.2, edgecolor=c, facecolor='none', zorder=2)
    ax.add_patch(rect)
    if bb_labels is not None:
      rect = Rectangle([bb[0], bb[1]-25], 90, 25, linewidth=0.2, edgecolor=c, facecolor=c, zorder=2)
      ax.add_patch(rect)
      plt.text(bb[0], bb[1], '%.3f' % bb_labels[i], size=0.2, zorder=2)
  os.makedirs('../reports/', exist_ok=True)
  fig.savefig('../reports/' + title + '.png', dpi=800)
  plt.show()
  