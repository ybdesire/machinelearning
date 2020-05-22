import pandas as pd
import numpy as np
import matplotlib.pylab as plt 

# read data
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] 
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1') 

# data merge
movie_ratings = pd.merge(movies, ratings) 
lens = pd.merge(movie_ratings, users)

# sort
# 用size( )函数得到每部电影的个数（即每部电影被评论的次数），按照从大到小排序，取最大的前20部电影列出如下：
most_rated = lens.groupby('title').size().sort_values(ascending=False)[:20]
print(most_rated)


'''equals sql

SELECT title, count(1)
FROM lens
GROUP BY title
ORDER BY 2 DESC
LIMIT 20;
'''