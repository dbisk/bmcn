# train.py
# training loop functions

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print("Using:", device)

  # send model to device
  model = model.to(device)

  # optimizer and loss function
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
  # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

  # training loop
  for epoch in range(epochs):
    # keep some statistics for each epoch
    running_loss = 0.0

    model.train() # set model to training mode
    for i, batch in enumerate(tqdm(train_dataloader)):
      inputs = batch['data']
      inputs = inputs.to(device)
      truths = batch['truth']
      truths = truths.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, truths)
      loss.backward()
      optimizer.step()

      # any statistics updates
      running_loss += loss.item()
    
    # normalize by number of batches and print
    running_loss /= len(train_dataloader)
    print("[%d] loss %.3f" % (epoch + 1, running_loss))
    
    # every few epochs, check the validation set
    # if (epoch % 5 == 0 or epoch + 1 == epochs):
    #   model.eval() # set model to eval mode
    #   with torch.no_grad():
    #     for batch in tqdm(test_dataloader): # TODO: consider wrapping in tqdm
    #       inputs = batch['data']
    #       inputs = inputs.to(device)

          # forward
          # outputs = model(inputs)
          # check correctness
          # calculate accuracy
    
    # step the lr scheduler
    scheduler.step()
  
  # training complete
  return model
