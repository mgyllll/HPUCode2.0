#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/4/27 8:48
# @Author : Luo Yong(MGYL)
# @File : test0427.py
# @Software: PyCharm

# 导入相关包
import pandas as pd
import numpy as np
import time
import math
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from itertools import combinations


def NetworkMappingProjection(df, path, per):
    if os.path.isfile(path + '/movies_network_u50.csv'):
        movies_net = pd.read_csv(path + '/movies_network_u50.csv')
        print('Yes')
    else:
        common_review_m = list()
        for i, j in tqdm(list(combinations(set(df['movie_id']), 2))):
            m1 = set(df[df['movie_id'] == i]['user_id'])
            m2 = set(df[df['movie_id'] == j]['user_id'])
            if m1 & m2:
                common_review_m.append([i, j, len(m1), len(m2), len(m1 & m2)])
        movies_net = pd.DataFrame(common_review_m, columns=['i', 'j', 'mi', 'mj', 'mij'])
        movies_net.to_csv(path + '/movies_network_u50.csv', index=False, encoding="utf_8_sig")

    if os.path.isfile(path + '/movies_network_weight_p' + str(int(100 * per)) + '_u50.csv'):
        movies_network_weight = pd.read_csv(path + '/movies_network_weight_p' + str(int(100 * per)) + '_u50.csv')
        print('Yes')
    else:
        movies_net['relation' + str(int(100 * per))] = movies_net.apply(
            lambda a: 1 if a['mij'] > per * min(a['mj'], a['mi']) else 0, axis=1)
        movies_network_weight = list()
        for p in tqdm(np.array(movies_net[movies_net['relation' + str(int(100 * per))] == 1])):
            m1 = set(df[df['movie_id'] == p[0]]['user_id'])
            m2 = set(df[df['movie_id'] == p[1]]['user_id'])
            tmp = 0
            mm = min(len(m1), len(m2))
            for l in m1 & m2:
                tmp += 1 / len(set(df[df['user_id'] == l]['movie_id']))
            movies_network_weight.append([p[0], p[1], tmp / mm])
        movies_network_weight = pd.DataFrame(movies_network_weight, columns=['i', 'j', 'wij'])
        movies_network_weight.to_csv(path + '/movies_network_weight_p' + str(int(100 * per)) + '_u50.csv', index=False,
                                     encoding="utf_8_sig")

    return movies_network_weight


def process_combinations(df, path, batch_size):
    comb_gen = combinations(set(df['movie_id']),2)
    batch = []
    ii = 0
    for i, c in enumerate(comb_gen):
        batch.append(c)
        if i % batch_size == 0 and i != 0:
            # 每次生成指定数量的组合后进行批处理
            batch_process(df, path, batch, ii)
            ii += 1
            batch = []

    # 处理最后一个批次
    if len(batch) > 0:
        batch_process(df, path, batch, ii)


def batch_process(df, path, batch, ii):
    common_review_m = list()
    for i,j in tqdm(batch):
        m1 = set(df[df['movie_id'] == i]['user_id'])
        m2 = set(df[df['movie_id'] == j]['user_id'])
        if m1 & m2:
            common_review_m.append([i, j, len(m1), len(m2), len(m1 & m2)])
    movies_net = pd.DataFrame(common_review_m, columns=['i', 'j', 'mi', 'mj', 'mij'])
    movies_net.to_csv(path+'/movies_network_u50_' + str(ii) + '.csv', index=False, encoding="utf_8_sig")
    # 执行批处理操作


if __name__ == '__main__':

    ratings_u50_douban = pd.read_csv('../data/2023data/douban/ratings_u50_douban.csv')
    movies_network_weight_ml_10m = NetworkMappingProjection(ratings_u50_douban, '../data/2023data/ml-10m', 0.8)