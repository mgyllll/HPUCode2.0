#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : movieLensProcess.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import functionX.algorithm as alg


def oscar_label_match(df, savefile):
    '''
    df的数据格式如下：
      movie_id           title                         genres
    0        1  Toy Story(1995)   Animation|Children's|Comedy
    1        2    Jumanji(1995)   Adventure|Children's|Fantas
    '''
    # 处理MovieLens数据集用于匹配获Oscar电影的数据集
    movies = df.copy(deep=True)  # 深拷贝
    movies['year'] = movies.apply(lambda a: a['title'].strip()[-5:-1], axis=1)
    movies['film'] = movies.apply(lambda a: a['title'].strip()[:-7], axis=1)
    movies['movie'] = movies.apply(lambda a: "".join(filter(str.isalnum, a['film'])).lower(), axis=1)
    movies.drop("genres", axis=1, inplace=True)
    movies.drop("title", axis=1, inplace=True)

    # 导入Oscars获奖电影数据集
    raw_oscars = pd.read_csv('./data/Oscars.csv')
    # # 去掉历届提名
    oscars = raw_oscars.drop(raw_oscars[raw_oscars['wonOscar'] == 0].index)
    # # 去掉非电影获奖奖项
    filt = ~oscars['award_Ch'].isin(['喜剧片最佳导演', '剧情片最佳导演', '最佳男主角', '最佳女主角', '最佳默片字幕对白', '最佳导演', '最佳男配角', '最佳女配角'])
    oscars = oscars[filt]
    # 喜剧片最佳导演 Best Director, Comedy Picture
    # 剧情片最佳导演 Best Director, Dramatic Picture
    # 最佳男主角 Best Performance by an Actor in a Leading Role
    # 最佳女主角 Best Performance by an Actress in a Leading Role
    # 最佳默片字幕对白 Best Writing, Title Writing
    # 最佳导演 Best Achievement in Directing
    # 最佳男配角 Best Performance by an Actor in a Supporting Role
    # 最佳女配角 Best Performance by an Actress in a Supporting Role
    # # 去重
    oscars['movie'] = oscars.apply(lambda a: "".join(filter(str.isalnum, a['movie_En'])).lower(), axis=1)
    oscars_list = set(np.array(oscars['movie']))
    movies['wonOscar'] = movies.apply(lambda a: 1 if a['movie'] in oscars_list else 0, axis=1)
    movies.drop("film", axis=1, inplace=True)
    movies.drop("movie", axis=1, inplace=True)
    movies.to_csv(savefile, index=False, encoding="utf_8_sig")
    return movies


def popularity_(df1, df2, day):
    ct = 0
    for m, _, _, f, l in np.array(df1):
        df = df2[(df2['movie_id'] == m) & (df2['timestamp'] <= (f + 24 * 60 * 60 * day))]
        ct += len(df)
    return ct / len(df1)


if __name__ == '__main__':
    print(1)
    # # # MovieLens数据加载
    # # 评分表ratings
    # rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    # raw_ratings = pd.read_table('../data/ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
    # # 电影表movies
    # mnames = ['movie_id', 'title', 'genres']
    # raw_movies = pd.read_table('../data/ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python',
    #                            encoding='ISO-8859-1')
    #
    # # # 数据预处理
    # # 筛选：每个用户至少包含50个评级，可变参数number of ratings = 50
    # ratings = raw_ratings[:]
    # ratings_by_user = ratings.groupby('user_id').size()  # 针对某一列进行计数count
    # active_user = ratings_by_user.index[ratings_by_user >= 50]
    # ratings_u50 = ratings[ratings.user_id.isin(active_user)]
    # ratings_u50.to_csv('../data/ml-1m/ratings_u50.csv', index=False, encoding="utf_8_sig")

    ratings_u50 = pd.read_csv('../data/ml-1m/ratings_u50.csv')

    # # 导入Oscar标注电影表
    # movies_label = pd.read_csv('../data/ml-1m/movies_label.csv')
    # # movies_label = oscar_label_match(raw_movies, './data/ml-1m/movies_label.csv')
    # # 过滤掉所有用户没有评级过的电影
    # movies_m = movies_label[movies_label.movie_id.isin(set(np.array(ratings_u50['movie_id'])))].copy()

    # # 记录每一部电影的最早和最晚评级时间
    # movies_m['firstTime'] = movies_m.apply(
    #     lambda m: ratings_u50[ratings_u50['movie_id'] == m['movie_id']]['timestamp'].min(), axis=1)
    # movies_m['lastTime'] = movies_m.apply(
    #     lambda m: ratings_u50[ratings_u50['movie_id'] == m['movie_id']]['timestamp'].max(), axis=1)

    # # 统计所有用户的打分最高分和最低分
    # u1 = list()
    # for u in set(np.array(ratings_u50['user_id'])):
    #     df = ratings_u50[ratings_u50['user_id'] == u].copy()
    #     u1.append([u, df['rating'].max(), df['rating'].min()])
    # users = pd.DataFrame(u1, columns=['user_id', 'maxR', 'minR'])

    # # # 感知力用户定义
    # # 计算每个用户的感知力值
    # thet = 0.3
    # users_D = list()
    # lenOscars = len(movies_m[movies_m['wonOscar'] == 1])
    # for u, M, _ in tqdm(np.array(users)):
    #     D_u = 0  # 记录用户感知高质量产品的数量
    #     d_u = 0  # 记录用户感知高质量产品且获Oscar奖的数量
    #     uu = np.array(users[users['user_id'] == u])
    #     for _, m, r, t in np.array(ratings_u50[ratings_u50['user_id'] == u]):
    #         mm = np.array(movies_m[movies_m['movie_id'] == m])
    #         if (r == uu[0][1]) & (t <= (mm[0][3] + thet * (mm[0][4] - mm[0][3]))):
    #             D_u += 1
    #             if mm[0][2]:
    #                 d_u += 1
    #     users_D.append([u, d_u, D_u, d_u / lenOscars])
    # users_P = pd.DataFrame(users_D, columns=['user_id', 'd', 'D', 'pty'])
    # users_P = users_P.sort_values(by='pty', axis=0, ascending=False)
    # users_P.to_csv('../data/ml-1m/users_perceptibility.csv', index=False, encoding="utf_8_sig")

    # users_P = pd.read_csv('../data/ml-1m/users_perceptibility.csv')
    # users_P = users_P.sort_values(by='pty', axis=0, ascending=False)

    # # # 先验实验
    # # 根据电影前10个高分记录统计的电影质量
    # movies_Q = list()
    # for m in tqdm(set(np.array(ratings_u50['movie_id']))):
    #     df = ratings_u50[(ratings_u50['movie_id'] == m) & (ratings_u50['rating'] == 5)].sort_values(
    #         by='timestamp', axis=0, ascending=True).head(10)
    #     movies_Q.append([m, users_P[users_P.user_id.isin(np.array(df['user_id']))]['pty'].mean() if len(df) > 1 else 0])
    # movies_Q = pd.DataFrame(movies_Q, columns=['movie_id', 'qty'])
    # movies_Q = movies_Q.sort_values(by='qty', axis=0, ascending=False)
    # movies_Q.to_csv('../data/ml-1m/movies_Quality.csv', index=False, encoding="utf_8_sig")

    # movies_Q = pd.read_csv('../data/ml-1m/movies_Quality.csv')
    # movies_Q = movies_Q.sort_values(by='qty', axis=0, ascending=False)

    # # 分析高质量产品和低质量产品随时间窗口的受欢迎程度
    # Q1 = movies_Q.head(int(len(movies_Q) * 0.1))
    # Q2 = movies_Q.tail(len(movies_Q) - int(len(movies_Q) * 0.1))
    # degree_Q = list()
    # degree_Q1 = list()
    # degree_Q2 = list()
    # for d in tqdm(range(1, 201)):
    #     degree_Q.append([d, popularity_(movies_m, ratings_u50, d)])
    #     degree_Q1.append([d, popularity_(movies_m[movies_m.movie_id.isin(np.array(Q1['movie_id']))], ratings_u50, d)])
    #     degree_Q2.append([d, popularity_(movies_m[movies_m.movie_id.isin(np.array(Q2['movie_id']))], ratings_u50, d)])
    # degree_Q = pd.DataFrame(degree_Q, columns=['t', 'avg_degree'])
    # degree_Q1 = pd.DataFrame(degree_Q1, columns=['t', 'avg_degree'])
    # degree_Q2 = pd.DataFrame(degree_Q2, columns=['t', 'avg_degree'])
    # degreeQ = pd.concat([degree_Q, degree_Q1, degree_Q2], axis=1)
    # degreeQ.columns = ['day', 'avgDegree_Q', 'd2', 'avgDegree_Q1', 'd3', 'avgDegree_Q2']
    # degreeQ.drop("d2", axis=1, inplace=True)
    # degreeQ.drop("d3", axis=1, inplace=True)
    # degreeQ.to_csv('./data/ml-1m/avgDegreeQ_30.csv', index=False, encoding="utf_8_sig")

    # # # 分析根据感知力所产生的高质量产品集合与低质量产品集合中Oscar获奖占比
    # oscar_Q = list()
    # oscar_Q1 = list()
    # oscar_Q2 = list()
    # for q in tqdm(range(5, 55, 5)):
    #     q1 = movies_Q.head(int(len(movies_Q) * q / 100))
    #     q2 = movies_Q.tail(len(movies_Q) - int(len(movies_Q) * q / 100))
    #     oscar_Q.append([q, len(movies_m[movies_m['wonOscar'] == 1]) / len(movies_m)])
    #     oscar_Q1.append([q, len(set(movies_m[movies_m['wonOscar'] == 1]['movie_id']) & set(q1['movie_id'])) / len(q1)])
    #     oscar_Q2.append([q, len(set(movies_m[movies_m['wonOscar'] == 1]['movie_id']) & set(q2['movie_id'])) / len(q2)])
    # oscar_Q = pd.DataFrame(oscar_Q, columns=['q', 'oscar_pro'])
    # oscar_Q1 = pd.DataFrame(oscar_Q1, columns=['q', 'oscar_pro'])
    # oscar_Q2 = pd.DataFrame(oscar_Q2, columns=['q', 'oscar_pro'])
    # oscarQ = pd.concat([oscar_Q, oscar_Q1, oscar_Q2], axis=1)
    # oscarQ.columns = ['q', 'oscar_Q', 'q2', 'oscar_Q1', 'q3', 'oscar_Q2']
    # oscarQ.drop("q2", axis=1, inplace=True)
    # oscarQ.drop("q3", axis=1, inplace=True)
    # oscarQ.to_csv('../data/ml-1m/wonOscarsQ_30.csv', index=False, encoding="utf_8_sig")

########################################################################################################################
    # degree_U = list()
    # degree_U1 = list()
    # degree_U2 = list()
    # for q in tqdm(range(5, 55, 5)):
    #     u1 = users_P.head(int(len(users_P) * q / 100))
    #     u2 = users_P.tail(len(users_P) - int(len(users_P) * q / 100))
    #     degree_U.append([q, len(ratings_u50) / len(users_P)])
    #     degree_U1.append([q, len(ratings_u50[ratings_u50.user_id.isin(np.array(u1['user_id']))]) / len(u1)])
    #     degree_U2.append([q, len(ratings_u50[ratings_u50.user_id.isin(np.array(u2['user_id']))]) / len(u2)])
    # degree_U = pd.DataFrame(degree_U, columns=['q', 'avg_degree'])
    # degree_U1 = pd.DataFrame(degree_U1, columns=['q', 'avg_degree'])
    # degree_U2 = pd.DataFrame(degree_U2, columns=['q', 'avg_degree'])
    # degreeU = pd.concat([degree_U, degree_U1, degree_U2], axis=1)
    # degreeU.columns = ['q', 'avgDegree_U', 'q2', 'avgDegree_U1', 'q3', 'avgDegree_U2']
    # degreeU.drop("q2", axis=1, inplace=True)
    # degreeU.drop("q3", axis=1, inplace=True)
    # degreeU.to_csv('../data/ml-1m/avgDegreeU_30.csv', index=False, encoding="utf_8_sig")

    # users_DFA = list()
    # for u in tqdm(set(np.array(users_P['user_id']))):
    #     ratings_u = ratings_u50[ratings_u50['user_id'] == u].copy()
    #     users_DFA.append([u, algorithm.func_DFA(np.array(ratings_u.sort_values(by='timestamp')['rating']), 1)])
    # users_DFA = pd.DataFrame(users_DFA, columns=['user_id', 'dfa'])
    # users_DFA['dfa'] = users_DFA['dfa'].fillna(0)
    # users_DFA.to_csv('../data/ml-1m/users_DFA.csv', index=False, encoding="utf_8_sig")
    # users_DFA = pd.read_csv('../data/ml-1m/users_DFA.csv')
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
    # dfaU.to_csv('../data/ml-1m/avgDfaU_30.csv', index=False, encoding="utf_8_sig")

    # test_ratings = ratings_u50.copy(deep=True)
    # test_ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
    # R, Q = alg.RBeta(test_ratings)

    # R = pd.read_csv('../data/ml-1m/BRreputations_u.csv')
    # rep_U = list()
    # rep_U1 = list()
    # rep_U2 = list()
    # for q in tqdm(range(5, 55, 5)):
    #     u1 = users_P.head(int(len(users_P) * q / 100))
    #     u2 = users_P.tail(len(users_P) - int(len(users_P) * q / 100))
    #     rep_U.append([q, R['R'].sum() / len(users_P)])
    #     rep_U1.append([q, R[R.userId.isin(np.array(u1['user_id']))]['R'].sum() / len(u1)])
    #     rep_U2.append([q, R[R.userId.isin(np.array(u2['user_id']))]['R'].sum() / len(u2)])
    # rep_U = pd.DataFrame(rep_U, columns=['q', 'avg_rep'])
    # rep_U1 = pd.DataFrame(rep_U1, columns=['q', 'avg_rep'])
    # rep_U2 = pd.DataFrame(rep_U2, columns=['q', 'avg_rep'])
    # repU = pd.concat([rep_U, rep_U1, rep_U2], axis=1)
    # repU.columns = ['q', 'avgRep_U', 'q2', 'avgRep_U1', 'q3', 'avgRep_U2']
    # repU.drop("q2", axis=1, inplace=True)
    # repU.drop("q3", axis=1, inplace=True)
    # repU.to_csv('../data/ml-1m/avgRepU_30.csv', index=False, encoding="utf_8_sig")
