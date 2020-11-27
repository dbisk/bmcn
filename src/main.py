# main.py
# filename subject to change

import numpy as np
from PIL import Image

import utils
import bmnn

def main():
  img = utils.load_img("./test_img/cman256.png")
  img = utils.add_noise(img, 20)

  blocks = bmnn.blockmatch(img, (128, 56))
  utils.show_blocks_on_image(img, blocks)
  # print("Exiting.")


if __name__ == "__main__":
  main()
