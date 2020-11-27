# bmnn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# some parameters
PATCH_SIZE = 8
SEARCH_SIZE = 39
THRESHOLD = 250,
MAX_BLOCKS = 32

class Block():
  def __init__(self, patch, x, y):
    self.x = x
    self.y = y
    self.data = patch

class BMNN(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, img):
    # create the denoised image we will return
    out = np.zeros(img.shape)
    # first, blockmatch each section of the image
    for x in range(0, img.shape[0] - PATCH_SIZE, PATCH_SIZE):
      for y in range(0, img.shape[1] - PATCH_SIZE, PATCH_SIZE):
        # find the matching blocks
        grp = blockmatch(img, (x, y))
        # throw the group through the network
        noise = 0 # TODO: replace with layers
        # subtract the learned noise to create the output
        out[x:x + PATCH_SIZE, y:y + PATCH_SIZE] = img[x:x + PATCH_SIZE, y:y + PATCH_SIZE] - noise

    return img

def blockmatch(img, coords, search_size=SEARCH_SIZE, patch_size=PATCH_SIZE, threshold=THRESHOLD, stride=1, N=MAX_BLOCKS):
  blocks = []
  patch_x = coords[0] # TODO: double check this, might be backwards
  patch_y = coords[1]
  start_x = 0 if patch_x - search_size/2 < 0 else int(patch_x - search_size/2)
  start_y = 0 if patch_y - search_size/2 < 0 else int(patch_y - search_size/2)
  end_x = img.shape[0] if patch_x + search_size/2 > img.shape[0] else int(patch_x + search_size/2)
  end_y = img.shape[1] if patch_y + search_size/2 > img.shape[1] else int(patch_y + search_size/2)

  base = img[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
  for x in range(start_x, end_x - patch_size, stride):
    for y in range(start_y, end_y - patch_size, stride):
      # grab each patch and compare to base patch
      patch = img[x:x + patch_size, y:y + patch_size]
      diff = np.linalg.norm(base - patch)
      if (diff < threshold):
        blocks.append(Block(patch, x, y))

  # catch the very odd case in which there are no matches and we accidentally skipped over the 
  # original patch itself
  if (len(blocks) == 0):
    blocks.append(Block(base, patch_x, patch_y))
  
  # remove random entries until we are at N blocks (might need to revisit this)
  while (len(blocks) > N):
    blocks.pop(np.random.randint(0, len(blocks)))

  # debug
  print("Found", len(blocks), "blocks.")
  return blocks

def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001):
  # TODO: this function needs to be heavily modified, currently just a skeleton
  # TODO: NOT FUNCTIONAL
  device = torch.device('cude') if torch.cuda.is_available() else torch.device('cpu')
  print("Using:", device)

  # send model to device
  model = model.to(device)

  # optimizer and loss function
  criterion = nn.CrossEntropyLoss()
  optimizer = optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

  # training loop
  for epoch in range(epochs):
    for i, batch in enumerate(train_dataloader): # TODO: consider wrapping in tqdm
      inputs = batch['data']
      inputs = inputs.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      # outputs = model(inputs)
      # loss = criterion(ouputs, labels)
      # loss.backward()
      # optimizer.step()

      # check whether our prediction was correct

      # any statistics updates
    
    # every few epochs, check the validation set
    if (epoch % 5 == 0 or epoch + 1 == epochs):
      model.eval() # set model to eval mode
      with torch.no_grad():
        for batch in test_dataloader: # TODO: consider wrapping in tqdm
          inputs = batch['data']
          inputs = inputs.to(device)

          # forward
          # outputs = model(inputs)
          # check correctness
          # calculate accuracy
    
    # step the lr scheduler
    scheduler.step()
  
  # training complete
  return model
