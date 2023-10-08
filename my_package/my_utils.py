#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : Utils.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import numpy as np
from tqdm import tqdm
import math
import torch
import pandas as pd
import random
import torch.optim as optim
import torch.nn as nn


def embeddings(df1, df2, df3, L):
    data_dict = {}
    for u in tqdm(set(np.array(df1['user_id']))):
        tmp_df = df1[df1['user_id'] == u].copy(deep=True)
        tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
        tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
        l = (L-1) if (L-1) < len(tmp_df) else len(tmp_df)
        movies_u = np.array(tmp_df['movie_id'].head(l))
        qty_m = list()  # RBeta电影的质量
        degree_m = list()  # 电影的度值
        for m in movies_u:
            qty_m.append(float(df2[df2['movie_id'] == m]['quality']))
            degree_m.append(len(df1[df1['movie_id'] == m]))
        A = np.zeros([L, L])
        A[0, 0] = df3[df3['user_id'] == u]['dfa']  # 用户DFA值
        ur = np.array(tmp_df['rating'].head(l))
        A[0, 1:l+1] = ur
        A[1:l+1, 0] = ur
        for i in range(1, l+1):
            A[i, i] = qty_m[i - 1]
        for r in range(1, l+1):
            for s in range(r + 1, l+1):
                A[r, s] = A[s, r] = 1 / math.exp(abs(abs(ur[r - 1] - ur[s - 1]) - abs(qty_m[r - 1] - qty_m[s - 1])))
        data_dict[u] = A
    return data_dict


def embeddings1(df1, df2, df3, L):
    data_dict = {}
    for u in tqdm(set(np.array(df1['user_id']))):
        tmp_df = df1[df1['user_id'] == u].copy(deep=True)
        tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
        tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
        l = (L-1) if (L-1) < len(tmp_df) else len(tmp_df)
        movies_u = np.array(tmp_df['movie_id'].head(l))
        qty_m = list()  # RBeta电影的质量
        degree_m = list()  # 电影的度值
        for m in movies_u:
            qty_m.append(float(df2[df2['movie_id'] == m]['Q']))
            degree_m.append(len(df1[df1['movie_id'] == m]))
        A = np.zeros([L, L])
        A[0, 0] = df3[df3['user_id'] == u]['dfa']  # 用户DFA值
        ur = np.array(tmp_df['rating'].head(l))
        A[0, 1:l+1] = ur
        A[1:l+1, 0] = ur
        data_dict[u] = A
    return data_dict


def tip(t, s):
    print(t * int(100 - 2 * len(s)), s, t * int(100 - 2 * len(s)))


# 训练模型
def train_model(loader, model, num_epochs, lr, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in range(num_epochs):
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.squeeze(labels)
            rs = model(data)
            _, predicted = torch.max(rs.data, 1)
            loss = criterion(rs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 100 == 0:
            print("Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def train_model1(loader, model, num_epochs, lr, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.CrossEntropyLoss()  # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in range(num_epochs):
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.squeeze(labels)
            # print('labels', labels)
            # print('data', data)
            rs = model(data)
            # print('rs', rs)
            # _, predicted = torch.max(rs.data, 1)
            onehot_target = torch.eye(2)[labels.long(), :]
            loss = criterion(rs, onehot_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 20 == 0:
            print(epoch, "Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def train_model_reg(loader, model, num_epochs, lr, path=None):
    """训练模型
        Parameters:
            loader: pytorch dataloader
            num_epochs: 训练的轮数
            lr:学习率
            path:模型存放路径
        return:
            model:训练好的模型
            loss_list:不同轮数的loss变化
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in tqdm(range(num_epochs)):
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            pred = model(data)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 50 == 0:
            print(epoch, "Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def undersampling(L, df_label, df1, df2, df3):

    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    num = min(len(pos_sample), len(neg_sample))

    pos_us = random.sample(list(np.array(pos_sample['user_id'])), num)
    neg_us = random.sample(list(np.array(neg_sample['user_id'])), num)
    pos_ratings = df2[df2.user_id.isin(pos_us)]
    neg_ratings = df2[df2.user_id.isin(neg_us)]
    pos_dict = embeddings(pos_ratings, df1, df3, L)
    neg_dict = embeddings(neg_ratings, df1, df3, L)
    pos_set = torch.empty(len(pos_dict), 1, L, L)
    for inx, matrix in enumerate(pos_dict.values()):
        pos_set[inx, :, :, :] = torch.from_numpy(matrix)
    # print(pos_set)
    pos_label = torch.empty(len(pos_dict), 1)
    for i in range(len(pos_dict)):
        pos_label[i, :] = 1
    # print(pos_label)
    neg_set = torch.empty(len(neg_dict), 1, L, L)
    for inx, matrix in enumerate(neg_dict.values()):
        neg_set[inx, :, :, :] = torch.from_numpy(matrix)
    # print(neg_set)
    neg_label = torch.empty(len(neg_dict), 1)
    for i in range(len(neg_dict)):
        neg_label[i, :] = 0
    # print(neg_label)
    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def sampling(L, df_label, df1, df2, df3):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    pos_num = len(pos_sample)
    neg_num = len(neg_sample)

    mm = min(pos_num, neg_num)
    total_num = max((pos_num+neg_num)//2, mm)

    if pos_num >= total_num:
        # print('随机取total_num个样本')
        pos_us = random.sample(list(np.array(pos_sample['user_id'])), total_num)
        pos_ratings = df2[df2.user_id.isin(pos_us)]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
    else:
        pos_ratings = df2[df2.user_id.isin(pos_sample['user_id'])]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_oversample = data_oversampling(pos_sample, L, total_num, df1, df2, df3)
        pos_dict.update(pos_oversample)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
        # print('随机生成total_num-pos_num')

    if neg_num >= total_num:
        # print('随机取total_num个样本')
        neg_us = random.sample(list(np.array(neg_sample['user_id'])), total_num)
        neg_ratings = df2[df2.user_id.isin(neg_us)]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        # print(neg_set)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
    else:
        neg_ratings = df2[df2.user_id.isin(neg_sample['user_id'])]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_oversample = data_oversampling(neg_sample, L, total_num, df1, df2, df3)
        neg_dict.update(neg_oversample)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
        # print('随机生成total_num-neg_num')

    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def oversampling(L, df_label, matrix_data, label, df1, df2, df3):

    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    num = max(len(pos_sample), len(neg_sample), 5000)

    torch_set = torch.empty(len(matrix_data), 1, L, L)
    for inx, matrix in enumerate(matrix_data.values()):
        torch_set[inx, :, :, :] = torch.from_numpy(matrix)
    torch_label = torch.empty(len(label), 1)
    for inx, v in enumerate(label.values()):
        torch_label[inx, :] = v
    pos_oversample = {}
    neg_oversample = {}
    if len(pos_sample) < num:
        pos_oversample = data_oversampling(pos_sample, L, num, df1, df2, df3)
        syn_pos_sample = torch.empty(num - len(pos_sample), 1, L, L)
        syn_pos_label = torch.empty(num - len(pos_sample), 1)
        for i in range(num - len(pos_sample)):
            syn_pos_sample[i, :, :, :] = torch.from_numpy(pos_oversample[i])
            syn_pos_label[i, :] = 1
        torch_set = torch.cat((torch_set, syn_pos_sample), 0)
        torch_label = torch.cat((torch_label, syn_pos_label), 0)
        # print('pos', len(pos_sample), '+', len(syn_pos_sample), '=', len(torch_set))
    if len(neg_sample) < num:
        neg_oversample = data_oversampling(neg_sample, L, num, df1, df2, df3)
        syn_neg_sample = torch.empty(num - len(neg_sample), 1, L, L)
        syn_neg_label = torch.empty(num - len(neg_sample), 1)
        for i in range(num - len(neg_sample)):
            syn_neg_sample[i, :, :, :] = torch.from_numpy(neg_oversample[i])
            syn_neg_label[i, :] = 0
        torch_set = torch.cat((torch_set, syn_neg_sample), 0)
        torch_label = torch.cat((torch_label, syn_neg_label), 0)
        # print('neg', len(neg_sample), '+', len(syn_neg_sample), '=', len(torch_label))

    return torch_set, torch_label


def data_oversampling(sample, L, num, df1, df2, df3):
    tip('*', 'Starting data oversampling ...')
    data_dict1 = {}
    sample_u = list(np.array(sample['user_id']))
    for i in tqdm(range(num - len(sample))):
        us = random.sample(sample_u, L - 1)
        um = list()
        ur = list()
        udfa = list()
        for j in range(L - 1):
            tmp_df = df2[df2['user_id'] == us[j]].copy(deep=True)
            tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
            tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
            movies_u = np.array(tmp_df['movie_id'].head(L - 1))
            if len(movies_u) <= j:
                um.append(movies_u[-1])
                ur.append(float(df2[(df2['movie_id'] == movies_u[-1]) & (df2['user_id'] == us[j])]['rating']))
            else:
                um.append(movies_u[j])
                ur.append(float(df2[(df2['movie_id'] == movies_u[j]) & (df2['user_id'] == us[j])]['rating']))
            udfa.append(float(df3[df3['user_id'] == us[j]]['dfa']))
        qty_m = list()  # RBeta电影的质量
        for m in um:
            qty_m.append(float(df1[df1['movie_id'] == m]['quality']))
        A = np.zeros([L, L])
        udfa = pd.DataFrame(udfa, columns=['dfa'])
        A[0, 0] = udfa['dfa'].mean()  # 用户DFA值
        A[0, 1:L] = ur
        A[1:L, 0] = ur
        for k in range(1, L):
            A[k, k] = qty_m[k - 1]
        for r in range(1, L):
            for s in range(r + 1, L):
                A[r, s] = A[s, r] = 1 / math.exp(abs(abs(ur[r - 1] - ur[s - 1]) - abs(qty_m[r - 1] - qty_m[s - 1])))
        data_dict1[i] = A
    tip('*', 'Data oversampling over!!')

    return data_dict1
