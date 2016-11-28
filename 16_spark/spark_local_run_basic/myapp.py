# !/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext


# 创建Spark Context。sc可以用于创建RDD
sc = SparkContext("local[2]", "First Spark App")# local[2] 说明本地模式，使用2个cpu。spark://host:port说明集群模式
# 解析cxv数据，作为RDD
data = sc.textFile('my_data.csv').map(lambda line:line.split(',')).map(lambda record: (record[0], record[1], record[2]))

# RDD的ACTION操作

num = data.count()# csv中数据行数
unique_users = data.map(lambda record: record[0]).distinct().count()# 第一列不相同的个数
total_revenue = data.map(lambda record: float(record[2])).sum()# 第三列的总和

# 最畅销的产品：购买人数最多的

products = data.map(lambda record: (record[1], 1.0)).reduceByKey(lambda a,b:a+b).collect()
most_popular = sorted(products, key=lambda x:x[1], reverse=True)[0]


print('total purchases: {0}'.format(num))
print('unique users: {0}'.format(unique_users))
print('total revenue: {0}'.format(total_revenue))
print('most popular product: {0}, with {1} purchases'.format(most_popular[0], most_popular[1]))




