#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : test.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-13
@Desc     : NULL
"""
import my_package.my_algorithm as alg
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    # ratings_u50 = pd.read_csv('./data/archive/ratings_u50.csv')
    users_P = pd.read_csv('./data/archive/users_perceptibility.csv')
    users_P = users_P.sort_values(by='pty', axis=0, ascending=False)

    # users_DFA = list()
    # for u in tqdm(set(np.array(users_P['user_id']))):
    #     ratings_u = ratings_u50[ratings_u50['user_id'] == u].copy()
    #     users_DFA.append([u, alg.func_DFA(np.array(ratings_u.sort_values(by='timestamp')['rating']), 1)])
    # users_DFA = pd.DataFrame(users_DFA, columns=['user_id', 'dfa'])
    # users_DFA['dfa'] = users_DFA['dfa'].fillna(0)
    # users_DFA.to_csv('./data/archive/users_DFA.csv', index=False, encoding="utf_8_sig")
    #
    # dfa_U = list()
    # dfa_U1 = list()
    # dfa_U2 = list()
    # for q in tqdm(range(5, 55, 5)):
    #     u1 = users_P.head(int(len(users_P) * q / 100))
    #     u2 = users_P.tail(len(users_P) - int(len(users_P) * q / 100))
    #     dfa_U.append([q, users_DFA['dfa'].sum() / len(users_P)])
    #     dfa_U1.append([q, users_DFA[users_DFA.user_id.isin(np.array(u1['user_id']))]['dfa'].sum() / len(u1)])
    #     dfa_U2.append([q, users_DFA[users_DFA.user_id.isin(np.array(u2['user_id']))]['dfa'].sum() / len(u2)])
    # dfa_U = pd.DataFrame(dfa_U, columns=['q', 'avg_dfa'])
    # dfa_U1 = pd.DataFrame(dfa_U1, columns=['q', 'avg_dfa'])
    # dfa_U2 = pd.DataFrame(dfa_U2, columns=['q', 'avg_dfa'])
    # dfaU = pd.concat([dfa_U, dfa_U1, dfa_U2], axis=1)
    # dfaU.columns = ['q', 'avgDfa_U', 'q2', 'avgDfa_U1', 'q3', 'avgDfa_U2']
    # dfaU.drop("q2", axis=1, inplace=True)
    # dfaU.drop("q3", axis=1, inplace=True)
    # dfaU.to_csv('./data/archive/avgDfaU_60.csv', index=False, encoding="utf_8_sig")

    # test_ratings = ratings_u50.copy(deep=True)
    # test_ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
    R, Q = alg.RBeta()

    rep_U = list()
    rep_U1 = list()
    rep_U2 = list()
    for q in tqdm(range(5, 55, 5)):
        u1 = users_P.head(int(len(users_P) * q / 100))
        u2 = users_P.tail(len(users_P) - int(len(users_P) * q / 100))
        rep_U.append([q, R['R'].sum() / len(users_P)])
        rep_U1.append([q, R[R.userId.isin(np.array(u1['user_id']))]['R'].sum() / len(u1)])
        rep_U2.append([q, R[R.userId.isin(np.array(u2['user_id']))]['R'].sum() / len(u2)])
    rep_U = pd.DataFrame(rep_U, columns=['q', 'avg_rep'])
    rep_U1 = pd.DataFrame(rep_U1, columns=['q', 'avg_rep'])
    rep_U2 = pd.DataFrame(rep_U2, columns=['q', 'avg_rep'])
    repU = pd.concat([rep_U, rep_U1, rep_U2], axis=1)
    repU.columns = ['q', 'avgRep_U', 'q2', 'avgRep_U1', 'q3', 'avgRep_U2']
    repU.drop("q2", axis=1, inplace=True)
    repU.drop("q3", axis=1, inplace=True)
    repU.to_csv('./data/archive/avgRepU_60.csv', index=False, encoding="utf_8_sig")
