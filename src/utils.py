# utils.py
# utility and helper functions for processing/prepping images

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def psnr_tensor(original, noisy):
  mse = torch.mean(torch.square(original - noisy))
  if (mse < 1e-3):
    return float("inf")
  return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr(original, noisy):
  mse = np.mean(np.square(original - noisy))
  if (mse < 1e-3):
    # PSNR blows up to infinity
    return float("inf")
  max_pixel = np.max(original)
  if (max_pixel > 1):
    max_pixel = 255.0
  else:
    max_pixel = 1.0
  
  return 20 * np.log10(max_pixel / np.sqrt(mse))

def add_noise(img, sigma = 30):
  noise = np.random.normal(0, sigma, img.shape).reshape(img.shape)
  return np.clip(img + noise, 0, 255) # only for grayscale, 0-1 images

def load_img(filepath, to_grayscale=False):
  img = Image.open(filepath)
  if (to_grayscale):
    img = img.convert(mode="L")
  return np.asarray(img)

def save_img(img, filepath, mode = "L"):
  Image.fromarray(img).convert(mode).save(filepath)

def show_blocks(blocks):
  """Shows the content of each individual block"""
  num_blocks = len(blocks)
  ncols = min(10, max(1, len(blocks))) # the max here just to catch 0s
  nrows = int(np.ceil(num_blocks / ncols))

  fig = plt.figure() # create the figure
  for index, b in enumerate(blocks, 1):
    # create a new subplot for our block image
    ax = fig.add_subplot(nrows, ncols, index)
    plt.imshow(b.data, cmap="gray") # add our image to the current axes
  
  plt.show() # show the graphic

def show_blocks_on_image(img, blocks=None):
  """Shows rectangles where the blocks are chosen on the actual image"""
  fig, ax = plt.subplots(1)
  ax.imshow(img, cmap="gray")
  if blocks is not None:
    for b in blocks:
      rect = patches.Rectangle((b.x, b.y), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
  
  plt.show() # show the graphic
