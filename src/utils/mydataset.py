import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class MyDataset():
  def __init__(self, mode='train', dataset='book', number=None):
    super(MyDataset, self).__init__()
    self.dataset = dataset
    self.number = number

    df = pd.read_csv('../data/' + dataset + '/ratings_final.txt',
                     sep='\t', header=None, index_col=None
                    ).values
    train, over = train_test_split(df, test_size=0.4, random_state=5555)

    test, valid = train_test_split(over, test_size=0.5, random_state=5555)
    self.item_enc = OneHotEncoder().fit([[i] for i in range(number[dataset]['entities'])])
    if mode == 'train':
        self.data = train
    elif mode == 'test':
        self.data = test
    else:
        self.data = valid
    logging.info(mode+' set size:'+str(self.data.shape[0]))

  def __getitem__(self, index):
      temp = self.data[index]
      # one hot initialization of the item
      item = self.item_enc.transform([[temp[1]]]).toarray()
      user, item, label = torch.tensor(temp[0], dtype=torch.long), torch.tensor(item, dtype=torch.float), torch.tensor(
          [temp[2]], dtype=torch.double)

      return user, item, label, torch.tensor(temp[1], dtype=torch.long)

  def __len__(self):
    return len(self.data)

########################
# train_loader function
########################
def data_loader(dataset, batch_size, number):
    train_loader = DataLoader(MyDataset(mode='train',dataset=dataset,number=number), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(mode='test',dataset=dataset,number=number), batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(MyDataset(mode='valid', dataset=dataset, number=number), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader
    
if __name__ == '__main__':
    DATASET = 'music'
    batch_size = 64
    number = {'music': {'users': 1872, 'items': 3846, 'interaction': 42346, 'entities': 9366, 'relations': 60, 'kg_triples': 15518, 'hyperbolicity': 0}}
    train_data, test_data, valid_data = data_loader(DATASET, batch_size, number)
    for user, item_hot, label, item in train_data:
      print(item)

