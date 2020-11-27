# bmcnn.py
# a similar structure to some bmcnn implementations around the internet

import torch.nn as nn

class BMCNN(nn.Module):
  def __init__(self, patch_depth, filter_size=24):
    super().__init__()

    # set up the layers
    self.conv1 = nn.Sequential(
      nn.Conv2d(patch_depth, 64, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1)),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(64, filter_size, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )
    self.conv3 = nn.Sequential(
      nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), padding=(3, 3), dilation=(3, 3)),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )
    self.conv4 = nn.Sequential(
      nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), padding=(4, 4), dilation=(4, 4)),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )
    self.conv5 = nn.Sequential(
      nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), padding=(3, 3), dilation=(3, 3)),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )
    self.conv6 = nn.Sequential(
      nn.Conv2d(filter_size, filter_size, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )
    self.conv7 = nn.Conv2d(filter_size, 1, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1))
  
  def forward(self, x):
    # x is expected to be of shape (-1, patch_depth, -1, -1)
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.conv7(out)
    return out # out should be of shape (-1, 1, -1, -1)
