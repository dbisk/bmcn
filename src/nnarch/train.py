# train.py
# training loop functions

import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001):
  # TODO: this function needs to be heavily modified, currently just a skeleton
  # TODO: NOT FUNCTIONAL
  device = torch.device('cude') if torch.cuda.is_available() else torch.device('cpu')
  print("Using:", device)

  # send model to device
  model = model.to(device)

  # optimizer and loss function
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
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
