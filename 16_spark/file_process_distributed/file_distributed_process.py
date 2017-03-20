# !/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.sql import SparkSession
import ftplib


spark = SparkSession\
        .builder\
        .appName("my data calc distributed")\
        .getOrCreate()


def myfun(m): # consider the parameter as each item of the rdd list
    path = 'dir_path'
    filename = m

    ftp = ftplib.FTP('10.xx.xx.xx')
    ftp.login("name", "password")
    ftp.cwd(path)
    ftp.retrbinary("RETR " + filename ,open(filename, 'wb').write)
    ftp.quit()

    num=0
    with open(filename, 'r') as f:
        num = int(f.readline())

    return num



# create spark context, which can be used to create RDD
sc = spark.sparkContext

# '123' in 's1.txt', '456' in 's2.txt', '789' in 's3.txt'
data_rdd = sc.parallelize(['s1.txt', 's2.txt', 's3.txt'])

result_rdd = data_rdd.map(myfun)

print(result_rdd.collect()) # output [123, 456, 789]

print('driver program stopped')
spark.stop()
