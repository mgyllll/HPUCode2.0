#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : main.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import my_package.my_utils as Utils
import my_package.my_models as Models
import my_package.my_algorithm as alg
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # 初始输入为 ratings_u50.csv movies_label.csv
    data_type = 'amazon'
    # 数据文件夹
    data_path = '../data/booksData/' + data_type
    # 实验日期
    exp_date = '20231007'
    # 实验文件夹
    file_path = data_path + '/exp' + exp_date
    # 使用os.makedirs()创建文件夹，如果路径不存在则递归创建
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 用户度值degree
    ratings_u50 = pd.read_csv(data_path + '/ratings_u50.csv')
    users_degree = ratings_u50.groupby('user_id').size().reset_index(name='degree')

    # 用户声誉reputation
    users_reputation, objects_quality = alg.createRBeta(ratings_u50, file_path, exp_date)

    # 用户DFA
    users_dfa = alg.createDFA(ratings_u50, file_path)

    df1 = pd.merge(users_degree, users_reputation, how='outer', on='user_id')
    df2 = pd.merge(df1, users_dfa, how='outer', on='user_id')

    # 用户感知力
    movies_label = pd.read_csv(data_path + '/movies_label.csv')
    users_perceptibility = alg.createPerceptibility(ratings_u50, movies_label, file_path, exp_date)

    # ##################################################卷积神经网络训练#####################################################
    # 参数设置
    seed = 100
    L = 28
    batch_size = 64
    num_epochs = 100
    lr = 0.0001  # 0.0005, 0.001, 0.005, 0.01, 0.05

    for q in range(5, 55, 5):
        print('parameter q:', q, '%')
        users_P1 = users_perceptibility.head(int(len(users_perceptibility) * (q / 100))).copy()
        users_P1['isHPU'] = 1
        users_P2 = users_perceptibility.tail(len(users_perceptibility) - int(len(users_perceptibility) * (q / 100))).copy()
        users_P2['isHPU'] = 0

        users_HPU = pd.concat([users_P1, users_P2])
        users_HPU.drop("d", axis=1, inplace=True)
        users_HPU.drop("D", axis=1, inplace=True)
        users_HPU.drop("perceptibility", axis=1, inplace=True)
        df = pd.merge(df2, users_HPU, how='right', on='user_id')

        y = df.isHPU
        x = df.drop(['user_id', 'isHPU'], axis=1)

        # 图表示学习：特征矩阵编码
        sampleX, labelY = Utils.sampling(L, df, objects_quality, ratings_u50, users_dfa)

        # 训练与测试数据集比例7:3
        xtrain1, xtest1, ytrain1, ytest1 = train_test_split(sampleX, labelY, test_size=0.3, random_state=seed)
        deal_data = TensorDataset(xtrain1, ytrain1)

        print(ytest1)

        PCNNList = list()
        for ep in range(5, 205, 5):
            pcnnList = list()
            print('parameter epoch:', ep)
            for i in range(5):  # 200
                Loader = DataLoader(dataset=deal_data, batch_size=batch_size, shuffle=True)
                rcnn = Models.CNN1(L)
                rcnn, loss = Utils.train_model1(Loader, rcnn, ep, lr, path=None)

                for j in range(10):
                    _, predicted2 = torch.max(rcnn(xtest1), 1)
                    labels2 = torch.squeeze(ytest1).long()
                    fpr, tpr, threshold = roc_curve(labels2, predicted2)  # 计算真正率和假正率
                    roc_auc = auc(fpr, tpr)  # 计算auc的值
                    pcnnList.append([precision_score(labels2, predicted2), recall_score(labels2, predicted2),
                                     f1_score(labels2, predicted2), roc_auc])
                    print('pcnn>>>>', precision_score(labels2, predicted2), recall_score(labels2, predicted2),
                          f1_score(labels2, predicted2), roc_auc)

            pcnnList = pd.DataFrame(pcnnList, columns=['precision', 'recall', 'f1', 'auc'])
            PCNNList.append([ep, pcnnList['precision'].mean(), pcnnList['recall'].mean(), pcnnList['f1'].mean(),
                             pcnnList['auc'].mean()])
        PCNNList = pd.DataFrame(PCNNList, columns=['ep', 'precision', 'recall', 'f1', 'auc'])
        PCNNList.to_csv(file_path + '/rs20231005_epochs_' + str(q) + '.csv', index=False, encoding="utf_8_sig")
