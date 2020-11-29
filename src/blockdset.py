# blockdset.py
# dataset class that premakes the matched blocks for network training

import os.path
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import bmnn

class BlockDataset(Dataset):
  def __init__(self, root_dir, sigma=40, transform=None):
    self.transform = transform
    self.data = []
    self.truths = []
    
    # load all the images immediately into a list
    # TODO: this will probably need to be revisited
    im_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
    imgs = []
    noisy = []
    for p in im_paths:
      im = np.asarray(Image.open(p).convert(mode="L"), dtype='float64')
      imgs.append(im)
      noise = np.random.normal(0, sigma, im.shape).reshape(im.shape)
      noisy.append(np.clip(im + noise, 0, 255))
    
    # blockmatch all the patches of every image
    # TODO: this probably takes a while. Needs to be changed.
    for i, im in enumerate(tqdm(noisy)):
      for x in range(0, im.shape[0] - bmnn.PATCH_SIZE + 1, bmnn.PATCH_SIZE):
        for y in range(0, im.shape[1] - bmnn.PATCH_SIZE + 1, bmnn.PATCH_SIZE):
          # get the ground truth
          gt = imgs[i][x:x + bmnn.PATCH_SIZE, y:y + bmnn.PATCH_SIZE]
          # find the matching blocks
          grp = bmnn.blockmatch(im, (x,y), stride=8)
          # turn the group into an array
          grp = bmnn.blocks_to_array(grp)
          # add this group to the actual data list
          self.data.append(grp / 255.0)
          self.truths.append(np.expand_dims(gt / 255.0, axis=0))

      
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    noisy = torch.Tensor(self.data[idx])
    truth = torch.Tensor(self.truths[idx])

    # currently not doing anything with transforms
    # TODO: how to transform both the noisy and clean imgs concurrently
    # if (self.transform):
    #   truth = self.transform(truth)
    #   noisy = self.transform(noisy)
    return {'data': noisy, 'truth': truth}