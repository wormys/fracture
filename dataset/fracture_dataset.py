"""
Data: 2021/05/10
Author: worith
Description: dataset
"""
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from utils.utils import cut_str
from config.config import global_config


class PDDataset(data.Dataset):
    def __init__(self, file_path, normalize=True):
        self.data = pd.read_csv(file_path)
        self.data.dropna(axis=0, how='any', inplace=True)
        self.in_feat = cut_str(global_config.get('feature', 'input_feat'))
        self.hidden_feat = cut_str(global_config.get('feature', 'hidden_feat'))
        self.out_feat = cut_str(global_config.get('feature', 'output_feat'))
        if normalize:
            self.data = self.data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            # self.data = self.data.apply(lambda x: (x - np.mean(x)) / (np.var(x) ))

    def __getitem__(self, index):
        x = torch.from_numpy(self.data.loc[index, self.in_feat].values).float()

        h = torch.from_numpy(self.data.loc[index, self.hidden_feat].values).float()

        y = torch.from_numpy(self.data.loc[index, self.out_feat].values).float()

        return x, h, y

    def __len__(self):
        return self.data.shape[0]




