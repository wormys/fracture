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
    def __init__(self, split_class, file_path, normalize=True):
        self.data = pd.read_csv(file_path)
        self.data.dropna(axis=0, how='any', inplace=True)
        if split_class == '2':
            self.in_feat = self.data.iloc[:, 0:5].columns.tolist()
            self.hidden_feat = self.data.iloc[:, 6:].columns.tolist()
        else:
            self.in_feat = self.data.iloc[:, 0:13].columns.tolist()
            self.hidden_feat = self.data.iloc[:, 14:].columns.tolist()
        self.out_feat = ['NPV']
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




