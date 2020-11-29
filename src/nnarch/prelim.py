# prelim.py
# a preliminary model to test basic functionality and baseline performance

import torch.nn as nn

class PrelimNN(nn.Module):
  def __init__(self, patch_size, patch_depth):
    super().__init__()
    self.patch_size = patch_size
    self.patch_depth = patch_depth
    self.conv = nn.Conv2d(patch_depth, 1, kernel_size=(3, 3), padding= (1, 1))

  def forward(self, grps):
    # grps is expected to be of shape (-1, patch_depth, 8, 8)
    out = self.conv(grps)
    return out # output should be of shape (-1, 1, 8, 8)
