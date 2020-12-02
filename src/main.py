# main.py
# filename subject to change

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import bmnn
# from bsdset import BSDataset
# from blockdset import BlockDataset
from stackedset import StackedDataset
# import nnarch.prelim
import nnarch.train
import nnarch.bmcnn

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
  tf = transforms.Compose([transforms.ToTensor()])
  trainset = StackedDataset(root_dir=str(Path('../data/train')), sigma=None, transform=tf)
  valset = StackedDataset(root_dir=str(Path('../data/val')), sigma=None, transform=tf)
  trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
  valloader = DataLoader(valset, batch_size=1, shuffle=True)

  # debug
  # img = next(iter(valloader))

  # model = nnarch.prelim.PrelimNN(8, 8)
  # output = bmnn.bmnn(torch.squeeze(img['data']), model)
  # print(img['data'].shape)

  # fig, ax = plt.subplots(1, 3)
  # ax[0].imshow(torch.squeeze(img['data'])[0], cmap="gray")
  # ax[0].set_title("Noisy Image")
  # ax[1].imshow(torch.squeeze(img['data'])[4], cmap="gray")
  # ax[1].set_title("Example channel 4")
  # ax[2].imshow(torch.squeeze(img['truth']), cmap="gray")
  # ax[2].set_title("Ground Truth")
  # plt.show()


  # begin training
  # model = nnarch.prelim.PrelimNN(8, 6)
  model = nnarch.bmcnn.BMCNN(6)
  # model.load_state_dict(torch.load("./test_model.pth"))
  model = nnarch.train.train(model, trainloader, valloader, epochs=55)
  torch.save(model.state_dict(), "./test_model.pth")
  model = model.to('cpu')

  # # TODO: temporarily just show an example full image
  img_true = utils.load_img("./test_img/peppers256.png")
  img = utils.add_noise(img_true, 30)
  model.eval()
  denoised = bmnn.bmnn(img, model, stride=2)
  print("PSNR:", utils.psnr(img_true / 255.0, denoised))
  fig, ax = plt.subplots(1, 3)
  ax[0].imshow(img_true, cmap='gray')
  ax[0].set_title("Clean Image")
  ax[1].imshow(img, cmap="gray")
  ax[1].set_title("Noisy Image")
  ax[2].imshow(denoised, cmap="gray")
  ax[2].set_title("Denoised")
  plt.show()

  # evaluate performance on test set

if __name__ == "__main__":
  main()
