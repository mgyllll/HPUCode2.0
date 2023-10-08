#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : priorDistribution.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    degreeQ = pd.read_csv('../data/archive/avgDegreeQ_60.csv')
    # degreeQ = degreeQ[degreeQ['day'] % 25 == 0]
    oscarQ = pd.read_csv('../data/archive/wonOscarsQ_60.csv')
    degreeU = pd.read_csv('../data/archive/avgDegreeU_60.csv')
    dfaU = pd.read_csv('../data/archive/avgDfaU_60.csv')
    repU = pd.read_csv('../data/archive/avgRepU_60.csv')

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(221)
    plt.plot(np.array(degreeQ['day']), np.array(degreeQ['avgDegree_Q']), color='k', linestyle='-', marker='v', label='Q')
    plt.plot(np.array(degreeQ['day']), np.array(degreeQ['avgDegree_Q1']), color='r', linestyle='-', marker='s', label='Q1')
    plt.plot(np.array(degreeQ['day']), np.array(degreeQ['avgDegree_Q2']), color='b', linestyle='-', marker='o', label='Q2')

    plt.subplot(222)
    plt.plot(np.array(oscarQ['q']), np.array(oscarQ['oscar_Q']), color='k', linestyle='-', marker='v', label='Q')
    plt.plot(np.array(oscarQ['q']), np.array(oscarQ['oscar_Q1']), color='r', linestyle='-', marker='s', label='Q1')
    plt.plot(np.array(oscarQ['q']), np.array(oscarQ['oscar_Q2']), color='b', linestyle='-', marker='o', label='Q2')
    plt.tight_layout()
    plt.legend(loc='center', bbox_to_anchor=(-.05, -0.2), ncol=3, fontsize=8)
    plt.savefig('../data/p_netflix_object.png')
    # plt.show()
########################################################################################################################
    plt.figure(figsize=(15, 8), dpi=300)
    plt.subplot(231)
    plt.plot(np.array(degreeU['q']), np.array(degreeU['avgDegree_U']), color='k', linestyle='-', marker='s', label='U')
    plt.plot(np.array(degreeU['q']), np.array(degreeU['avgDegree_U1']), color='r', linestyle='-', marker='o', label='U1')
    plt.plot(np.array(degreeU['q']), np.array(degreeU['avgDegree_U2']), color='b', linestyle='-', marker='v', label='U2')

    plt.subplot(232)
    plt.plot(np.array(dfaU['q']), np.array(dfaU['avgDfa_U']), color='k', linestyle='-', marker='s', label='U')
    plt.plot(np.array(dfaU['q']), np.array(dfaU['avgDfa_U1']), color='r', linestyle='-', marker='o', label='U1')
    plt.plot(np.array(dfaU['q']), np.array(dfaU['avgDfa_U2']), color='b', linestyle='-', marker='v', label='U2')

    plt.subplot(233)
    plt.plot(np.array(repU['q']), np.array(repU['avgRep_U']), color='k', linestyle='-', marker='s', label='U')
    plt.plot(np.array(repU['q']), np.array(repU['avgRep_U1']), color='r', linestyle='-', marker='o', label='U1')
    plt.plot(np.array(repU['q']), np.array(repU['avgRep_U2']), color='b', linestyle='-', marker='v', label='U2')
    plt.tight_layout()
    plt.legend(loc='center', bbox_to_anchor=(-0.6, -0.2), ncol=3, fontsize=10)
    plt.savefig('../data/p_netflix_users.png')
    # plt.show()