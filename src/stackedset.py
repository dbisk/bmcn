# stackedset.py
# represents a dataset of a full image, stacked with its blocks

import os.path
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import bmnn

class StackedDataset(Dataset):
  def __init__(self, root_dir, sigma=40, patch_depth=5, transform=None):
    self.transform = transform
    
    # load all the images immediately into a list
    # TODO: this will probably need to be revisited
    im_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
    imgs = []
    noisy = []
    for p in im_paths:
      im = np.asarray(Image.open(p).convert(mode="L"), dtype='float64')
      imgs.append(im)
      # add noise
      n_sigma = np.random.randint(5, 100) if sigma is None else sigma
      noise = np.random.normal(0, n_sigma, im.shape).reshape(im.shape)
      noisy.append(np.clip(im + noise, 0, 255))
    
    self.data = []
    self.truths = []
    
    # blockmatch all the patches of every image
    # TODO: this takes a while, probably needs to be changed
    for i, im in enumerate(tqdm(noisy)):
      # create the data image
      full_img = np.zeros((patch_depth + 1, im.shape[0], im.shape[1]), dtype=im.dtype)
      full_img[0] = im
      for x in range(0, im.shape[0] - bmnn.PATCH_SIZE + 1, bmnn.PATCH_SIZE):
        for y in range(0, im.shape[1] - bmnn.PATCH_SIZE + 1, bmnn.PATCH_SIZE):
          # find the matching blocks
          grp = bmnn.blockmatch(im, (x,y), stride=2)
          # turn the noisy group into an array
          grp = bmnn.blocks_to_array(grp)
          # place that group into the stack below the area
          full_img[1:, x:x + bmnn.PATCH_SIZE, y:y + bmnn.PATCH_SIZE] = grp
      
      # append this full image to the data array
      self.data.append(full_img / 255.0)
      self.truths.append(np.expand_dims(imgs[i], 0) / 255.0)

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
          