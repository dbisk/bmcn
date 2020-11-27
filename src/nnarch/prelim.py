# prelim.py
# a preliminary model to test basic functionality and baseline performance

import torch.nn as nn

class PrelimNN(nn.Module):
  def __init__(self, patch_size, patch_depth):
    super().__init__()
    self.patch_size = patch_size
    self.patch_depth = patch_depth
    self.conv1 = nn.Conv2d(patch_depth, 64, kernel_size=(3, 3), padding=(1, 1))
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
    self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
    self.conv4 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

    self.relu = nn.ReLU(inplace=True)
    self.norm32 = nn.BatchNorm2d(32)
    self.norm64 = nn.BatchNorm2d(64)

  def forward(self, grps):
    # grps is expected to be of shape (-1, patch_depth, patch_size, patch_size)
    out = self.conv1(grps)
    out = self.relu(self.norm64(out))
    out = self.conv2(out)
    out = self.relu(self.norm64(out))
    out = self.conv3(out)
    out = self.relu(self.norm32(out))
    out = self.conv4(out)
    return out # output should be of shape (-1, 1, patch_size, patch_size)
