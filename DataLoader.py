from Config import configs
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy

class ItemData(Dataset):
    def __init__(self, path):
        self.dataset = pd.read_csv(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index,0:8].values
        label = self.dataset.iloc[index,8:8+configs['pred_step']].values
        return {
            "data": torch.from_numpy(data.astype(int)).long(),
            "label": torch.from_numpy(label.astype(int)).long()
        }


Train_DataLoader = DataLoader(ItemData(configs['train_path']), batch_size=configs['batch_size'], shuffle=False)
Val_DataLoader = DataLoader(ItemData(configs['val_path']), batch_size=configs['batch_size'], shuffle=False)


