#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : algorithm.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd


def DFA(data, ni, fittime):
    n = len(data) // ni
    nf = int(n * ni)

    n_mean = np.mean(data[:nf])
    y = []
    y_hat = []
    for i in range(nf):
        y.append(np.sum(data[:i + 1] - n_mean))
    for i in range(int(n)):
        x = np.arange(1, ni + 1, 1)
        y_temp = y[int(i * ni + 1) - 1:int((i + 1) * ni)]
        coef = np.polyfit(x, y_temp, deg=fittime)
        y_hat.append(np.polyval(coef, x))
    fn = np.sqrt(sum((np.asarray(y) - np.asarray(y_hat).reshape(-1)) ** 2) / nf)

    return fn


def func_DFA(rs, deg):
    n = []
    for i in np.arange(5, len(rs) // 4 + 1, 1):
        n.append(i)
    f = []
    for i in range(len(n)):
        f.append(DFA(rs, n[i], deg))
    coef = np.polyfit(np.log10(n), np.log10(f), deg)
    y_hat = np.polyval(coef, np.log10(n))
    for i in range(len(n)):
        y_hat[i] = 10 ** y_hat[i]
    #     plt.plot(n,y_hat,c='r')
    #     plt.scatter(n, f, c='b')
    #     plt.xlabel('n')
    #     plt.ylabel('F(n)')
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.show()

    return coef[0]


def tip(t, s):
    print(t * int(50 - 2 * len(s)), s, t * int(50 - 2 * len(s)))


def createRBeta(df, fpath, edate):
    # 深拷贝一份数据
    raw_ratings = df.copy()

    filepath = fpath + '/tmp' + edate

    # 使用os.makedirs()创建文件夹，如果路径不存在则递归创建
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # 第一步，将评分标准化为[-1, 1]区间
    tip('*', '第一步，将评分标准化为[-1, 1]区间')
    if not os.path.exists(filepath + '/tmp_ratings_N.csv'):
        rating_N = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'rating_N'])
        for u in tqdm(set(raw_ratings['user_id'])):
            rs = raw_ratings[raw_ratings['user_id'] == u].copy()
            maxR = rs['rating'].max()
            minr = rs['rating'].min()
            rs['rating_N'] = rs.apply(lambda r: 2 * (r['rating'] - minr) / (maxR - minr) - 1 if (maxR != minr) else 0,
                                      axis=1)
            rating_N = pd.concat([rating_N, rs])
        rating_N.to_csv(filepath + '/tmp_ratings_N.csv', index=False, encoding="utf_8_sig")
    else:
        rating_N = pd.read_csv(filepath + '/tmp_ratings_N.csv')

    # 第二步，将标准化的数值转化为花哨程度
    tip('*', '第二步，将标准化的数值转化为花哨程度')
    if not os.path.exists(filepath + '/tmp_ratings_F.csv'):
        rating_F = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'rating_N', 'polarity'])
        for m in tqdm(set(rating_N['movie_id'])):
            rs1 = rating_N[rating_N['movie_id'] == m].copy()
            plus = len(rating_N[(rating_N['movie_id'] == m) & (rating_N['rating_N'] >= 0)]['rating_N'])
            minus = len(rating_N[(rating_N['movie_id'] == m) & (rating_N['rating_N'] < 0)]['rating_N'])
            rs1['polarity'] = rs1.apply(lambda a: 1.0 if plus >= minus else -1.0, axis=1)
            rating_F = pd.concat([rating_F, rs1])
        rating_F['rating_F'] = rating_F.apply(lambda a: 1.0 if (a['rating_N'] * a['polarity']) >= 0 else -1.0, axis=1)
        rating_F.to_csv(filepath + '/tmp_ratings_F.csv', index=False, encoding="utf_8_sig")
    else:
        rating_F = pd.read_csv(filepath + '/tmp_ratings_F.csv')

    # 第三步，计算用户声誉值R_i
    tip('*', '第三步，计算用户声誉值R_i')
    if not os.path.exists(fpath + '/users_reputation.csv'):
        users_reputation = list()
        for u in tqdm(set(np.array(rating_F['user_id']))):
            s = len(rating_F[(rating_F['user_id'] == u) & (rating_F['rating_F'] == 1.0)])
            f = len(rating_F[(rating_F['user_id'] == u) & (rating_F['rating_F'] == -1.0)])
            users_reputation.append([u, (s + 1) / (s + f + 2)])
        users_reputation = pd.DataFrame(users_reputation, columns=['user_id', 'reputation'])
        users_reputation.to_csv(fpath + '/users_reputation.csv', index=False, encoding="utf_8_sig")
    else:
        users_reputation = pd.read_csv(fpath + '/users_reputation.csv')

    # 第四步，计算对象质量值
    tip('*', '第四步，计算对象质量值')
    if not os.path.exists(filepath + '/tmp_objects_quality.csv'):
        tmp = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'rating_N', 'polarity', 'rating_F', 'R_r'])
        for u in tqdm(set(np.array(rating_F['user_id']))):
            rs2 = rating_F[rating_F['user_id'] == u].copy()
            rep = users_reputation[users_reputation['user_id'] == u]['reputation']
            rs2['R_r'] = rs2.apply(lambda a: a['rating'] * rep, axis=1)
            tmp = pd.concat([tmp, rs2])
        tmp.to_csv(filepath + '/tmp_objects_quality.csv', index=False, encoding="utf_8_sig")
    else:
        tmp = pd.read_csv(filepath + '/tmp_objects_quality.csv')

    if not os.path.exists(fpath + '/objects_quality.csv'):
        objects_quality = list()
        for o in tqdm(set(np.array(tmp['movie_id']))):
            # 选择产品α的所有用户i
            o_ratings = tmp[tmp['movie_id'] == o].copy()
            o_users = set(np.array(o_ratings['user_id']))
            numerator = o_ratings[o_ratings.user_id.isin(o_users)]['R_r'].sum()
            df = users_reputation[users_reputation.user_id.isin(o_users)].copy()
            denominator = df['reputation'].sum()
            max_R = df['reputation'].max()
            objects_quality.append([o, float(max_R * numerator / denominator)])
        objects_quality = pd.DataFrame(objects_quality, columns=['movie_id', 'quality'])
        objects_quality.to_csv(fpath + '/objects_quality.csv', index=False, encoding="utf_8_sig")
    else:
        objects_quality = pd.read_csv(fpath + '/objects_quality.csv')

    return users_reputation, objects_quality


def createDFA(df, fpath):
    # 深拷贝一份数据
    raw_ratings = df.copy()

    if not os.path.exists(fpath + '/users_dfa.csv'):
        users_dfa = list()
        for u in tqdm(set((raw_ratings['user_id']))):
            ratings_u = raw_ratings[raw_ratings['user_id'] == u].copy()
            users_dfa.append([u, func_DFA(np.array(ratings_u.sort_values(by='timestamp')['rating']), 1)])
        users_dfa = pd.DataFrame(users_dfa, columns=['user_id', 'dfa'])
        users_dfa['dfa'] = users_dfa['dfa'].fillna(0)
        users_dfa.to_csv(fpath + '/users_dfa.csv', index=False, encoding="utf_8_sig")
    else:
        users_dfa = pd.read_csv(fpath + '/users_dfa.csv')
    users_dfa.fillna(0, inplace=True)

    return users_dfa


def createPerceptibility(ratings_u50, movies_label, file_path, edate):
    filepath = file_path + '/tmp' + edate

    if not os.path.exists(filepath + '/tmp_movies_m.csv'):
        # 过滤掉所有用户没有评级过的电影
        movies_m = movies_label[movies_label['movie_id'].isin(set(ratings_u50['movie_id']))].copy()

        # 记录每一部电影的最早和最晚评级时间
        tqdm.pandas()
        movies_m['firstTime'] = movies_m.progress_apply(
            lambda m: ratings_u50[ratings_u50['movie_id'] == m['movie_id']]['timestamp'].min(), axis=1)
        movies_m['lastTime'] = movies_m.progress_apply(
            lambda m: ratings_u50[ratings_u50['movie_id'] == m['movie_id']]['timestamp'].max(), axis=1)

        movies_m.to_csv(filepath + '/tmp_movies_m.csv', index=False, encoding="utf_8_sig")
    else:
        movies_m = pd.read_csv(filepath + '/tmp_movies_m.csv')

    if not os.path.exists(filepath + '/tmp_users_info.csv'):
        # 统计所有用户的打分最高分和最低分
        users_info = list()
        for u in tqdm(set(ratings_u50['user_id'])):
            df = ratings_u50[ratings_u50['user_id'] == u].copy()
            users_info.append([u, df['rating'].max(), df['rating'].min()])
        users_info = pd.DataFrame(users_info, columns=['user_id', 'maxR', 'minR'])
        users_info.to_csv(filepath + '/tmp_users_info.csv', index=False, encoding="utf_8_sig")
    else:
        users_info = pd.read_csv(filepath + '/tmp_users_info.csv')

    if not os.path.exists(file_path + '/users_perceptibility.csv'):
        # #感知力用户定义
        # 计算每个用户的感知力值
        thet = 0.3
        users_D = list()
        lenOscars = len(movies_m[movies_m['awardee'] == 1])
        for idx, row in tqdm(users_info.iterrows()):
            D_u = 0  # 记录用户认定为高质量产品的数量
            d_u = 0  # 记录用户认定为高质量产品且获Oscar奖的数量，即真实高质量产品
            for _, r in ratings_u50[ratings_u50['user_id'] == row['user_id']].iterrows():
                mm = movies_m[movies_m['movie_id'] == r['movie_id']].iloc[0]
                if r['rating'] == row['maxR'] and r['timestamp'] <= (mm['firstTime'] + thet * (mm['lastTime'] - mm['firstTime'])):
                    D_u += 1
                    if mm['awardee']:
                        d_u += 1
            users_D.append([row['user_id'], d_u, D_u, d_u / lenOscars])
        users_perceptibility = pd.DataFrame(users_D, columns=['user_id', 'd', 'D', 'perceptibility'])
        users_perceptibility = users_perceptibility.sort_values(by='perceptibility', axis=0, ascending=False)
        users_perceptibility.to_csv(file_path + '/users_perceptibility.csv', index=False, encoding="utf_8_sig")
    else:
        users_perceptibility = pd.read_csv(file_path + '/users_perceptibility.csv')
        users_perceptibility = users_perceptibility.sort_values(by='perceptibility', axis=0, ascending=False)

    return users_perceptibility

