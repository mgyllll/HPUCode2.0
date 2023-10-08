#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : plotting.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-08
@Desc     : NULL
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style('ticks')

if __name__ == '__main__':
    PCNNList = pd.read_csv('./data/result/PCNNdata0925.csv')
    GBMList = pd.read_csv('./data/result/GBMdata0925.csv')
    RFList = pd.read_csv('./data/result/RFdata0925.csv')
    SVMList = pd.read_csv('./data/result/SVMdata0925.csv')

    # visualize results
    plt.figure(figsize=(18, 10), dpi=120)

    plt.subplot(231)
    plt.plot(np.array(PCNNList['q']), np.array(PCNNList['precision']), markersize=8, color='r', linestyle='-',
             marker='s', label='PCNN')
    plt.plot(np.array(GBMList['q']), np.array(GBMList['precision']), markersize=8, color='g', linestyle='-', marker='o',
             label='GBM')
    plt.plot(np.array(RFList['q']), np.array(RFList['precision']), markersize=8, color='b', linestyle='-', marker='v',
             label='RF')
    plt.plot(np.array(SVMList['q']), np.array(SVMList['precision']), markersize=8, color='k', linestyle='-', marker='*',
             label='SVM')
    plt.ylabel(r'precision', fontsize=16, fontweight='bold')
    plt.xlabel(r'$q$', fontsize=16, fontweight='bold')
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.xticks(np.arange(0, 60, 10), ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=12)
    plt.text(3, 0.94, '(a)', fontsize=14, fontweight='bold')

    plt.subplot(232)
    plt.plot(np.array(PCNNList['q']), np.array(PCNNList['recall']), markersize=8, color='r', linestyle='-', marker='s',
             label='PCNN')
    plt.plot(np.array(GBMList['q']), np.array(GBMList['recall']), markersize=8, color='g', linestyle='-', marker='o',
             label='GBM')
    plt.plot(np.array(RFList['q']), np.array(RFList['recall']), markersize=8, color='b', linestyle='-', marker='v',
             label='RF')
    plt.plot(np.array(SVMList['q']), np.array(SVMList['recall']), markersize=8, color='k', linestyle='-', marker='*',
             label='SVM')
    plt.ylabel(r'$recall$', fontsize=16, fontweight='bold')
    plt.xlabel(r'$q$', fontsize=16, fontweight='bold')
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.xticks(np.arange(0, 60, 10), ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=12)
    plt.text(3, 0.94, '(b)', fontsize=14, fontweight='bold')

    plt.subplot(233)
    plt.plot(np.array(PCNNList['q']), np.array(PCNNList['f1']), markersize=8, color='r', linestyle='-', marker='s',
             label='PCNN')
    plt.plot(np.array(GBMList['q']), np.array(GBMList['f1']), markersize=8, color='g', linestyle='-', marker='o',
             label='GBM')
    plt.plot(np.array(RFList['q']), np.array(RFList['f1']), markersize=8, color='b', linestyle='-', marker='v',
             label='RF')
    plt.plot(np.array(SVMList['q']), np.array(SVMList['f1']), markersize=8, color='k', linestyle='-', marker='*',
             label='SVM')
    plt.ylabel(r'f1-measure', fontsize=16, fontweight='bold')
    plt.xlabel(r'$q$', fontsize=16, fontweight='bold')
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.xticks(np.arange(0, 60, 10), ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=12)
    plt.text(3, 0.94, '(c)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(-0.10, -0.15), ncol=4, fontsize=14)
    plt.savefig('./data/result/netflix_alg0928-1.png')
    plt.show()
