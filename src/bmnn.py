# bmnn.py

import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# some parameters
PATCH_SIZE = 8
SEARCH_SIZE = 39
THRESHOLD = 1300,
MAX_BLOCKS = 12

class Block():
  def __init__(self, patch, x, y):
    self.x = x
    self.y = y
    self.data = patch
  
  def __lt__(self, other):
    return self.x < other.x
  
  def __gt__(self, other):
    return self.x > other.x

def blocks_to_array(blocks):
  out = np.zeros((len(blocks), PATCH_SIZE, PATCH_SIZE))
  for idx, blk in enumerate(blocks):
    out[idx] = blk.data
  return out

def get_clean_blocks(img, blocks):
  out = np.zeros((len(blocks), PATCH_SIZE, PATCH_SIZE))
  for idx, blk in enumerate(blocks):
    out[idx] = img[blk.x:blk.x + PATCH_SIZE, blk.y:blk.y + PATCH_SIZE]
  return out

def bmnn(img, model, stride=1):
  # create the denoised image we will return
  full_img = np.zeros((MAX_BLOCKS + 1, img.shape[0], img.shape[1]), dtype=img.dtype)
  full_img[0] = img
  # first, blockmatch each section of the image
  for x in range(0, img.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
    for y in range(0, img.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
      # find the matching blocks
      grp = blockmatch(img, (x, y), stride=stride)
      # turn the group into an array
      grp = blocks_to_array(grp)
      # place that group into the stack below the area
      full_img[1:, x:x + PATCH_SIZE, y:y + PATCH_SIZE] = grp

  # throw the stacked image through the network
  full_img = torch.Tensor(full_img / 255.0)
  with torch.no_grad():
    out = model(torch.unsqueeze(full_img, 0))
  return torch.squeeze(out).numpy()

def blockmatch(img, coords, search_size=SEARCH_SIZE, patch_size=PATCH_SIZE, threshold=THRESHOLD, stride=1, N=MAX_BLOCKS, stats=None):
  blocks = []
  patch_x = coords[0]
  patch_y = coords[1]
  start_x = 0 if patch_x - search_size/2 < 0 else int(patch_x - search_size/2)
  start_y = 0 if patch_y - search_size/2 < 0 else int(patch_y - search_size/2)
  end_x = img.shape[0] if patch_x + search_size/2 > img.shape[0] else int(patch_x + search_size/2)
  end_y = img.shape[1] if patch_y + search_size/2 > img.shape[1] else int(patch_y + search_size/2)

  base = img[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
  # TODO: can check for a faster BM algorithm than brute force
  for x in range(start_x, end_x - patch_size, stride):
    for y in range(start_y, end_y - patch_size, stride):
      # grab each patch and compare to base patch
      patch = img[x:x + patch_size, y:y + patch_size]
      mse = np.mean(np.square(base - patch))
      if (mse < threshold):
        heapq.heappush(blocks, (mse, Block(patch, x, y)))
        # debug
        # TODO: change the way this is done, currently needs stats to be a list due to
        # python not having a way to pass by reference
        if stats is not None:
          stats[0] += 1

  # add copies of the original patch until we are at N blocks (TODO: revisit this)
  while (len(blocks) < N):
    heapq.heappush(blocks, (0, Block(base, patch_x, patch_y)))
  
  # take the top N blocks
  result = []
  for i in range(N):
    result.append(heapq.heappop(blocks)[1])

  return result

