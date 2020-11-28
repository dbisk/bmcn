# prelim.py
# a preliminary model to test basic functionality and baseline performance

import torch.nn as nn

class PrelimNN(nn.Module):
  def __init__(self, patch_size, patch_depth):
    super().__init__()
    self.patch_size = patch_size
    self.patch_depth = patch_depth

  def forward(self, grps):
    return grps
