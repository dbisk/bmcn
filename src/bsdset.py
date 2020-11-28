# bsdset.py
# dataset class to load the Berkeley Segmentation Dataset images

import os.path
import glob
from torch.utils.data import Dataset
from utils import add_noise, load_img

class BSDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.im_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
    self.transform = transform
  
  def __len__(self):
    return len(self.im_paths)
  
  def __getitem__(self, idx):
    # TODO: allow slices instead of forcing just 1 idx at a time
    truth = load_img(self.im_paths[idx], to_grayscale=True)
    if (self.transform):
      truth = self.transform(truth)
    noisy = add_noise(truth)
    return {'data': noisy, 'truth': truth}
