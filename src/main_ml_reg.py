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

    type = 'ml-1m'
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
    seed = 10
    batch_size = 64
    num_epochs = 200  # 200
    lr = 0.001  # MovieLens->0.0005
    # movieLen_data = Utils.embeddings(ratings_u50, Rquality_m, users_DFA, L)

    users_P = pd.read_csv('../data/moviesData/' + type + '/users_perceptibility.csv')
    users_P = users_P.sort_values(by='pty', axis=0, ascending=False)
    users_HPU = users_P[:]
    users_HPU.drop("d", axis=1, inplace=True)
    users_HPU.drop("D", axis=1, inplace=True)
    label = dict(np.array(users_HPU[['user_id', 'pty']]))
    for l in range(24, 68, 4):
        PCNNList = list()
        pcnnList = list()
        print('parameter L:', l)
        movieLen_data = Utils.embeddings1(ratings_u50, Rquality_m, users_DFA, l)
        torch_set = torch.empty(len(movieLen_data), 1, l, l)
        for inx, matrix in enumerate(movieLen_data.values()):
            torch_set[inx, :, :, :] = torch.from_numpy(matrix)
        torch_label = torch.empty(len(label), 1)
        for inx, v in enumerate(label.values()):
            torch_label[inx, :] = v
        xtrain1, xtest1, ytrain1, ytest1 = train_test_split(torch_set, torch_label, test_size=0.3, random_state=seed)
        PCNNList.append(ytest1.detach().numpy().reshape(-1).tolist())
        for i in range(1):  # 200
            deal_data = TensorDataset(xtrain1, ytrain1)
            Loader = DataLoader(dataset=deal_data, batch_size=batch_size, shuffle=True)
            rcnn = Models.CNN_Reg(l)
            rcnn, loss = Utils.train_model_reg(Loader, rcnn, num_epochs, lr, path=None)

            for j in range(5):
                tt = rcnn(xtest1).detach().numpy().reshape(-1).tolist()
                pcnnList.append(tt)
        pcnnList = np.array(pcnnList)
        PCNNList.append(pcnnList.mean(axis=0).tolist())
        str1 = pd.Series(PCNNList[0])
        str2 = pd.Series(PCNNList[1])
        print(str1.corr(str2, method='kendall'))
        PCNNList = pd.DataFrame({
            0: PCNNList[0],
            l: PCNNList[1]
        })

        # PCNNList.to_csv('../data/moviesData/result/BiNCNN1028_reg_ml_' + str(l) + '.csv', index=False, encoding="utf_8_sig")
