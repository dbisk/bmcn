# main.py
# filename subject to change

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import bmnn
from bsdset import BSDataset
import nnarch.prelim

def main():
  # TODO: use argparse to make this way easier to use (low priority)
  # img = utils.load_img("./test_img/cman256.png")
  # img = utils.add_noise(img, 20)

  # blocks = bmnn.blockmatch(img, (128, 56))
  # utils.show_blocks_on_image(img, blocks)
  # print("Exiting.")

  # testing
  # root_dir = str(Path('../data/val'))
  # bsds = BSDataset(root_dir=root_dir)
  
  # print("Dataset size:", len(bsds))
  # idx = np.random.randint(0, len(bsds))
  # img = bsds[idx]
  
  # show images together
  # fig, ax = plt.subplots(1, 2)
  # ax[0].imshow(img['data'], cmap="gray")
  # ax[0].set_title("Noisy Image")
  # ax[1].imshow(img['truth'], cmap="gray")
  # ax[1].set_title("Ground Truth")
  # plt.show()

  # actual code
  trainset = BSDataset(root_dir=str(Path('../data/train')))
  valset = BSDataset(root_dir=str(Path('../data/val')))
  trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
  valloader = DataLoader(valset, batch_size=1, shuffle=True)

  # debug
  img = next(iter(trainloader))

  model = nnarch.prelim.PrelimNN(8, 32)
  output = bmnn.bmnn(img['data'][0], model)

  fig, ax = plt.subplots(1, 3)
  ax[0].imshow(img['data'][0], cmap="gray")
  ax[0].set_title("Noisy Image")
  ax[1].imshow(img['truth'][0], cmap="gray")
  ax[1].set_title("Ground Truth")
  ax[2].imshow(output, cmap="gray")
  ax[2].set_title("BMNN")
  plt.show()


  # begin training
  # model = nnarch.prelim.PrelimNN(8, 32)
  # model = nnarch.train.train(model, trainloader, valloader)

  # evaluate performance on test set

if __name__ == "__main__":
  main()
