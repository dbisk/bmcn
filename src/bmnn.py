# bmnn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# some parameters
PATCH_SIZE = 8
SEARCH_SIZE = 39
THRESHOLD = 250,
MAX_BLOCKS = 32

class Block():
  def __init__(self, patch, x, y):
    self.x = x
    self.y = y
    self.data = patch

def _blocks_to_array(blocks):
  out = np.zeros((len(blocks), PATCH_SIZE, PATCH_SIZE))
  for idx, blk in enumerate(blocks):
    out[idx] = blk.data
  return out

def bmnn(img, model):
  # create the denoised image we will return
  out = np.zeros(img.shape)
  # first, blockmatch each section of the image
  for x in range(0, img.shape[0] - PATCH_SIZE, PATCH_SIZE):
    for y in range(0, img.shape[1] - PATCH_SIZE, PATCH_SIZE):
      # find the matching blocks
      grp = blockmatch(img, (x, y))
      # turn the group into a tensor
      grp = _blocks_to_array(grp)
      # throw the group through the network
      with torch.no_grad():
        out_patch = model(torch.unsqueeze(torch.from_numpy(grp), 0).float())
      # place that patch back into the output
      out[x:x + PATCH_SIZE, y:y + PATCH_SIZE] = out_patch[0,0,:,:]

  return out

def blockmatch(img, coords, search_size=SEARCH_SIZE, patch_size=PATCH_SIZE, threshold=THRESHOLD, stride=1, N=MAX_BLOCKS):
  blocks = []
  patch_x = coords[0] # TODO: double check this, might be backwards
  patch_y = coords[1]
  start_x = 0 if patch_x - search_size/2 < 0 else int(patch_x - search_size/2)
  start_y = 0 if patch_y - search_size/2 < 0 else int(patch_y - search_size/2)
  end_x = img.shape[0] if patch_x + search_size/2 > img.shape[0] else int(patch_x + search_size/2)
  end_y = img.shape[1] if patch_y + search_size/2 > img.shape[1] else int(patch_y + search_size/2)

  base = img[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
  for x in range(start_x, end_x - patch_size, stride):
    for y in range(start_y, end_y - patch_size, stride):
      # grab each patch and compare to base patch
      patch = img[x:x + patch_size, y:y + patch_size]
      diff = np.linalg.norm(base - patch)
      if (diff < threshold):
        blocks.append(Block(patch, x, y))

  # catch the very odd case in which there are no matches and we accidentally skipped over the 
  # original patch itself
  if (len(blocks) == 0):
    blocks.append(Block(base, patch_x, patch_y))

  # add copies of the original patch until we are at N blocks (TODO: revisit this)
  while (len(blocks) < N):
    blocks.append(Block(base, patch_x, patch_y))
  
  # remove random entries until we are at N blocks (TODO: revisit this)
  while (len(blocks) > N):
    blocks.pop(np.random.randint(0, len(blocks)))

  return blocks

