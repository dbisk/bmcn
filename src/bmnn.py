# bmnn.py

from torch import nn

class BMNN(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, grps):
    return grps

  def match(self, img):
    return img