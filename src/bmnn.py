# bmnn.py

import numpy as np
from torch import nn

class Block():
  def __init__(self, patch, x, y):
    self.x = x
    self.y = y
    self.data = patch

class BMNN(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, grps):
    return grps

  def blockmatch(self, img, coords, search_size=39, patch_size=8, threshold=250, stride=1, N=32):
    blocks = []
    patch_x = coords[0] # TODO: double check this, might be backwards
    patch_y = coords[1]
    start_x = 0 if patch_x - search_size / 2 < 0 else int(patch_x - search_size / 2)
    start_y = 0 if patch_y - search_size / 2 < 0 else int(patch_y - search_size / 2)
    end_x = img.shape[0] if patch_x + search_size / 2 > img.shape[0] else int(patch_x + search_size / 2)
    end_y = img.shape[1] if patch_y + search_size / 2 > img.shape[1] else int(patch_y + search_size / 2)

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
    
    # remove random entries until we are at N blocks (might need to revisit this)
    while (len(blocks) > N):
      blocks.pop(np.random.randint(0, len(blocks)))

    # debug
    print("Found", len(blocks), "blocks.")
    return blocks
