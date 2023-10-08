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
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import my_package.my_utils as Utils
import my_package.my_models as Models
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    type = 'ml_1m'
    # 用户度值
    ratings_u50 = pd.read_csv('../data/moviesData/' + type + '/ratings_u50.csv')
    users_D = ratings_u50['user_id'].value_counts().reset_index()
    users_D.columns = ['user_id', 'degree']
    # 用户声誉
    users_R = pd.read_csv('../data/moviesData/' + type + '/BRreputations_u.csv')
    users_R.columns = ['user_id', 'rep']
    # 用户DFA
    users_DFA = pd.read_csv('../data/moviesData/' + type + '/users_DFA.csv')
    users_DFA['dfa'] = users_DFA['dfa'].fillna(0)  # 将值为NAN的填充为0

    df1 = pd.merge(users_D, users_R, how='outer', on='user_id')
    df2 = pd.merge(df1, users_DFA, how='outer', on='user_id')

    Rquality_m = pd.read_csv('../data/moviesData/' + type + '/BRquality_o.csv')

    # 卷积神经网络训练
    seed = 100
    L = 28
    batch_size = 64
    num_epochs = 100  # 200
    lr = 0.0001  # MovieLens->0.0005
    # movieLen_data = Utils.embeddings(ratings_u50, Rquality_m, users_DFA, L)

    users_P = pd.read_csv('../data/moviesData/' + type + '/users_perceptibility.csv')
    users_P = users_P.sort_values(by='pty', axis=0, ascending=False)
    for q in range(5, 55, 5):
        print('parameter q:', q, '%')
        users_P1 = users_P.head(int(len(users_P) * (q / 100)))
        users_P1['isHPU'] = True
        users_P2 = users_P.tail(len(users_P) - int(len(users_P) * (q / 100)))
        users_P2['isHPU'] = False
        users_HPU = users_P1.append(users_P2)
        users_HPU.drop("d", axis=1, inplace=True)
        users_HPU.drop("D", axis=1, inplace=True)
        users_HPU.drop("pty", axis=1, inplace=True)
        df = pd.merge(df2, users_HPU, how='right', on='user_id')
        df.isHPU = df.isHPU.astype(str).map({'False': 0, 'True': 1})

        df = df.sort_values(by='user_id', axis=0, ascending=True)
        label = dict(np.array(df[['user_id', 'isHPU']]))

        y = df.isHPU
        x = df.drop(['user_id', 'isHPU'], axis=1)

        movieLen_data = Utils.embeddings(ratings_u50, Rquality_m, users_DFA, L)
        sampleX, labelY = Utils.oversampling(L, df, movieLen_data, label, Rquality_m, ratings_u50, users_DFA)
        # sampleX, labelY = Utils.undersampling(l, df, Rquality_m, ratings_u50, users_DFA)
        xtrain1, xtest1, ytrain1, ytest1 = train_test_split(sampleX, labelY, test_size=0.3, random_state=seed)
        deal_data = TensorDataset(xtrain1, ytrain1)

        PCNNList = list()
        for ep in range(5, 205, 5):
            pcnnList = list()
            print('parameter epoch:', ep)
            for i in range(2):  # 200
                Loader = DataLoader(dataset=deal_data, batch_size=batch_size, shuffle=True)
                rcnn = Models.CNN1(L)
                rcnn, loss = Utils.train_model1(Loader, rcnn, ep, lr, path=None)

                for j in range(5):
                    _, predicted2 = torch.max(rcnn(xtest1), 1)
                    labels2 = torch.squeeze(ytest1).long()
                    fpr, tpr, threshold = roc_curve(labels2, predicted2)  # 计算真正率和假正率
                    roc_auc = auc(fpr, tpr)  # 计算auc的值
                    pcnnList.append([precision_score(labels2, predicted2), recall_score(labels2, predicted2),
                                     f1_score(labels2, predicted2), roc_auc])
                    print('pcnn>>>>', precision_score(labels2, predicted2), recall_score(labels2, predicted2),
                          f1_score(labels2, predicted2), roc_auc)

            pcnnList = pd.DataFrame(pcnnList, columns=['precision', 'recall', 'f1', 'auc'])
            PCNNList.append([ep, pcnnList['precision'].mean(), pcnnList['recall'].mean(), pcnnList['f1'].mean(), pcnnList['auc'].mean()])
        PCNNList = pd.DataFrame(PCNNList, columns=['ep', 'precision', 'recall', 'f1', 'auc'])
        PCNNList.to_csv('../data/moviesData/result/BiNCNN20231003_ml_' + str(q) + '.csv', index=False, encoding="utf_8_sig")
