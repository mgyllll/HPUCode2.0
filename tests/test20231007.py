#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/10/7 10:41
# @Author : Luo Yong(MGYL)
# @File : test20231007.py
# @Software: PyCharm


import pandas as pd

# 初始输入为 ratings_u50.csv movies_label.csv
data_type = 'amazon'
# 数据文件夹
data_path = '../data/booksData/' + data_type

ratings_u50 = pd.read_csv(data_path + '/ratings_u50.csv')
ratings_u50['movie_id'] = ratings_u50['movie_id'].astype(str)
movies_label = pd.read_csv(data_path + '/movies_label.csv')
movies_label['movie_id'] = movies_label['movie_id'].astype(str)
movies_m = movies_label[movies_label['movie_id'].isin(set(ratings_u50['movie_id']))].copy()

print(len(set(ratings_u50['movie_id'])))
print(len(set(movies_label['movie_id'])))
print(len(set(movies_m['movie_id'])))

print(len(set(ratings_u50['movie_id']) & set(movies_label['movie_id'])))

ratings1 = ratings_u50[['user_id', 'movie_id']].copy()
ratings1.drop_duplicates(inplace=True)
ratings2 = ratings_u50.drop_duplicates()
print('用户-电影评分数据量')
print(len(ratings_u50))
print(len(ratings1))
print(len(ratings2))
# 使用groupby对'列1'和'列2'进行分组，然后对'列3'进行字符连接
result = ratings_u50.groupby(['user_id', 'movie_id'])['rating'].apply(lambda x: ';'.join(str(x))).reset_index()

# 打印结果
print(result)
