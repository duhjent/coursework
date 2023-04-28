from torch import nn
import torch
import copy
import numpy as np

def train_model(net, train_iter, test_iter, loss, trainer, num_epochs, device='cpu'):
  train_losses = []
  test_losses = []
  num_batches = len(train_iter)
  best_loss = 1e10
  best_model_wts = copy.deepcopy(net.state_dict())

  for epoch in range(num_epochs):
    print(f'started epoch {epoch+1}/{num_epochs}')
    batch_losses = []
    net.train()
    for i, (X, y) in enumerate(train_iter):
      X, y = X.to(device), y.to(device)
      trainer.zero_grad()
      y_pred = net(X)
      l = loss(y_pred, y)
      batch_losses.append(l.item())
      l.backward()
      trainer.step()
      if (i+1)%20 == 0:
        print(f'\tbatch {i + 1}/{num_batches} loss: {np.mean(batch_losses)}')
    train_losses.append(np.mean(batch_losses))
    
    batch_losses = []
    net.eval()
    with torch.no_grad():
      for X, y in test_iter:
        y_pred = net(X)
        l = loss(y_pred, y)
        batch_losses.append(l.item())
    test_losses.append(np.mean(batch_losses))
    print(f'epoch {epoch+1}/{num_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}')
    if best_loss > test_losses[-1]:
      best_loss = test_losses[-1]
      best_model_wts = copy.deepcopy(net.state_dict())
  
  net.load_state_dict(best_model_wts)
  return train_losses, test_losses

class RNNModel(nn.Module):
  def __init__(self, seq_len, hidden_size, num_layers, dropout_prob=0.2, bidirectional=True, num_hidden_neutrons=None, recur_ctor=nn.LSTM):
    super(RNNModel, self).__init__()
    self.input_size = seq_len
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.rnn = recur_ctor(1, num_layers=num_layers, dropout=dropout_prob, hidden_size=hidden_size, bidirectional=bidirectional)
    fc_in_size = 4 * hidden_size if bidirectional else hidden_size
    if num_hidden_neutrons is None or (isinstance(num_hidden_neutrons, list) and len(num_hidden_neutrons) == 0):
      self.fc = nn.Linear(fc_in_size, 1)
    else:
      out_layers = []
      out_layers.append(nn.Linear(fc_in_size, num_hidden_neutrons[0]))
      for i, n in enumerate(num_hidden_neutrons[1:]):
        out_layers.append(nn.ReLU())
        out_layers.append(nn.Linear(num_hidden_neutrons[i], n))
      out_layers.append(nn.ReLU())
      out_layers.append(nn.Linear(num_hidden_neutrons[-1], 1))
      
      self.fc = nn.Sequential(*out_layers)
  def forward(self, x):
    out, _ = self.rnn(x.transpose(0 ,1))
    if self.bidirectional:
      out = torch.cat([out[0], out[-1]], dim=1)
    else:
      out = out[-1]
    return self.fc(out)
