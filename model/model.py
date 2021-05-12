"""
Date: 2021/05/10
Author: worith
"""

import torch
import torch.nn.functional as F


class NetX2Y(torch.nn.Module):
    def __init__(self, is_physical_info, n_feature, n_output):
        super(NetX2Y, self).__init__()

        self.hidden1 = torch.nn.Linear(n_feature, 20)   # 隐藏层线性输出
        self.is_physical_info = is_physical_info
        self.hidden2 = torch.nn.Linear(20, 40)  # 隐藏层线性输出
        self.hidden3 = torch.nn.Linear(40, 20)  # 隐藏层线性输出
        if self.is_physical_info:
            self.predict = torch.nn.Linear(40, n_output)   # 输出层线性输出
        else:
            self.predict = torch.nn.Linear(20, n_output)
        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        if self.is_physical_info:
            x = torch.cat([x, self.physical_info], 1)
        x = self.predict(x)             # 输出值
        return x

    def add_physical_info(self, physical_info):
        self.physical_info = physical_info


class NetH2Y(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(NetH2Y, self).__init__()

        self.hidden1 = torch.nn.Linear(n_feature, 20)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(20, 40)  # 隐藏层线性输出
        self.hidden3 = torch.nn.Linear(40, 20)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(20, n_output)   # 输出层线性输出
        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        physical_info = x

        x = self.predict(x)             # 输出值
        if self.training:
            return x
        else:
            return x, physical_info
