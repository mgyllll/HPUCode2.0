#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : auc.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-29
@Desc     : NULL
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import functionX.Utils as Utils
import functionX.Models as Models
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    type = 'archive'
    # 用户度值
    ratings_u50 = pd.read_csv('./data/' + type + '/ratings_u50.csv')
    users_D = ratings_u50['user_id'].value_counts().reset_index()
    users_D.columns = ['user_id', 'degree']
    # 用户声誉
    users_R = pd.read_csv('./data/' + type + '/BRreputations_u.csv')
    users_R.columns = ['user_id', 'rep']
    # 用户DFA
    users_DFA = pd.read_csv('./data/' + type + '/users_DFA.csv')
    users_DFA['dfa'] = users_DFA['dfa'].fillna(0)  # 将值为NAN的填充为0

    df1 = pd.merge(users_D, users_R, how='outer', on='user_id')
    df2 = pd.merge(df1, users_DFA, how='outer', on='user_id')

    Rquality_m = pd.read_csv('./data/' + type + '/BRquality_o.csv')

    # 卷积神经网络训练
    seed = 100
    L = 28
    batch_size = 64
    num_epochs = 100  # Netflix->100 MovieLens->200
    lr = 0.0001  # Netflix->0.0001 MovieLens->0.0005

    GBMList = list()
    SVMList = list()
    RFList = list()
    PCNNList = list()
    users_P = pd.read_csv('./data/' + type + '/users_perceptibility.csv')
    users_P = users_P.sort_values(by='pty', axis=0, ascending=False)
    for l in range(4, 68, 4):
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

            for i in range(1):  # 200
                sampleX, labelY = Utils.undersampling(L, df, Rquality_m, ratings_u50, users_DFA)
                xtrain1, xtest1, ytrain1, ytest1 = train_test_split(sampleX, labelY, test_size=0.3, random_state=seed)
                deal_data = TensorDataset(xtrain1, ytrain1)
                Loader = DataLoader(dataset=deal_data, batch_size=batch_size, shuffle=True)
                rcnn = Models.CNN1(L)
                rcnn, loss = Utils.train_model1(Loader, rcnn, num_epochs, lr, path=None)

                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed)
                gbc = GradientBoostingClassifier().fit(xtrain, ytrain)
                rfc = RandomForestClassifier(class_weight='balanced').fit(xtrain, ytrain)
                svmc = svm.SVC(class_weight='balanced').fit(xtrain, ytrain)

                pre_gbc = gbc.predict(xtest)
                proba_gbc = gbc.predict_proba(xtest)
                print(111111111, proba_gbc)
                fpr_1, tpr_1, threshold_1 = roc_curve(ytest, pre_gbc)  # 计算真正率和假正率
                roc_auc_1 = auc(fpr_1, tpr_1)  # 计算auc的值

                pre_rfc = rfc.predict(xtest)
                proba_rfc = rfc.predict_proba(xtest)
                print(22222222, proba_rfc)
                fpr_2, tpr_2, threshold_2 = roc_curve(ytest, pre_rfc)  # 计算真正率和假正率
                roc_auc_2 = auc(fpr_2, tpr_2)  # 计算auc的值

                pre_svmc = svmc.predict(xtest)
                proba_svmc = svmc.predict_proba(xtest)
                print(3333333333, proba_svmc)
                fpr_3, tpr_3, threshold_3 = roc_curve(ytest, pre_svmc)  # 计算真正率和假正率
                roc_auc_3 = auc(fpr_3, tpr_3)  # 计算auc的值

                _, predicted2 = torch.max(rcnn(xtest1), 1)
                print(4444444444, rcnn(xtest1))
                labels2 = torch.squeeze(ytest1).long()
                fpr_4, tpr_4, threshold_4 = roc_curve(ytest1, predicted2)  # 计算真正率和假正率
                roc_auc_4 = auc(fpr_4, tpr_4)  # 计算auc的值
                print(roc_auc_score(labels2, predicted2))
                print(roc_curve(ytest1, predicted2))

                plt.figure(figsize=(8, 5))
                plt.plot(fpr_1, tpr_1, color='darkorange',  # 假正率为横坐标，真正率为纵坐标做曲线
                         lw=2, label='Extreme Gradient Boosting (area = %0.3f)' % roc_auc_1, linestyle='-')  # linestyle
                # 为线条的风格（共五种）,color为线条颜色
                plt.plot(fpr_2, tpr_2, color='red',
                         lw=2, label='Random Forest (area = %0.3f)' % roc_auc_2, linestyle='--')
                plt.plot(fpr_3, tpr_3, color='green',
                         lw=2, label='Support Vector Machine (area = %0.3f)' % roc_auc_3, linestyle='--')
                plt.plot(fpr_4, tpr_4, color='#800080',
                         lw=2, label='BiNCNN (area = %0.3f)' % roc_auc_4, linestyle=':')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([-0.02, 1.05])  # 横竖增加一点长度 以便更好观察图像
                plt.ylim([-0.02, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.savefig("./data/hyh_" + str(q) + ".png", dpi=600)  # 保存图片，dpi设置分辨率
                plt.show()
