#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : Models.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN1(nn.Module):
    def __init__(self, L):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * int(L / 4) * int(L / 4), 256)  # 28->32*4*4
        self.fc2 = nn.Linear(256, 2)
        self.dropout1 = nn.Dropout(0.5)  # N-0.5  M-0.5
        self.dropout2 = nn.Dropout(0.6)  # N-0.6  M-0.6

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.size())
        # x = x.view(-1, 32*4*4)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)  # 0.5
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # 0.6
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, L):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*4*4, 256)  # 28->32*4*4
        self.fc2 = nn.Linear(256, 2)
        self.dropout1 = nn.Dropout(0.5)  # N-0.3  M-0.5
        self.dropout2 = nn.Dropout(0.6)  # N-0.4  M-0.6

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.size())
        x = x.view(-1, 32*4*4)
        x = self.dropout1(x)  # 0.5
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # 0.6
        x = self.fc2(x)
        return x


class CNN_Reg(nn.Module):
    def __init__(self, L):
        super(CNN_Reg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(128 * int(L / 16) * int(L / 16), 200)  # 28->32*4*4
        self.fc2 = nn.Linear(200, 10)  # 28->32*4*4
        self.fc3 = nn.Linear(10, 1)  # 28->32*4*4

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # x = torch.flatten(x, 1)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x