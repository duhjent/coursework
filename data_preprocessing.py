import time
import datetime
import pandas as pd
import requests
import shutil
import numpy as np
import torch.utils.data as torch_data
import torch

def to_sequences(data, seq_len):
  res = []
  for i in range(len(data) - seq_len):
    res.append(data[i:i+seq_len])
  
  return np.array(res)

def split_seq(data, seq_len, train_percent):
  assert train_percent > 0 and train_percent < 1
  seq_data = to_sequences(data, seq_len)
  
  num_train = int(len(seq_data) * train_percent)
  
  X_train = seq_data[:num_train, :-1]
  y_train = seq_data[:num_train, -1, :]
  
  X_test = seq_data[num_train:, :-1]
  y_test = seq_data[num_train:, -1, :]
  
  return X_train, y_train, X_test, y_test

def prepare_data(data, seq_len, train_percent, batch_size):
  X_train, y_train, X_test, y_test = split_seq(data, seq_len, train_percent)

  train_dataset = torch_data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
  train_iter = torch_data.DataLoader(train_dataset, batch_size=batch_size)

  test_dataset = torch_data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
  test_iter = torch_data.DataLoader(train_dataset, batch_size=batch_size)

  return train_iter, test_iter

def get_download_url(currency_name):
  start_time = int(time.mktime(datetime.datetime(2012, 1, 1).timetuple()))
  end_time = int(time.mktime(datetime.datetime.utcnow().timetuple()))
  return f'https://query1.finance.yahoo.com/v7/finance/download/{currency_name}?period1={start_time}&period2={end_time}&interval=1d&events=history'

def get_data(currency_name):
  file_name = f'./data/{currency_name}.csv'
  headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:98.0) Gecko/20100101 Firefox/98.0'}
  url = get_download_url(currency_name)
  with requests.get(url, stream=True, headers=headers) as r:
    with open(file_name, 'wb') as f:
      shutil.copyfileobj(r.raw, f)
  
  df = pd.read_csv(file_name)
  df['Date'] = df['Date'].astype('datetime64')
  return df
