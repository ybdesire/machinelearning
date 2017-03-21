# !/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
from subprocess import check_output


spark = SparkSession\
        .builder\
        .appName("get slave address")\
        .getOrCreate()


def myfun(m): # consider the parameter as each item of the rdd list
    ips = check_output(['hostname', '--all-ip-addresses'])# get slave IP
    dir_path = os.path.dirname(os.path.realpath(__file__))# get slave current working directory
    return (ips,dir_path)



# create spark context, which can be used to create RDD
sc = spark.sparkContext

# '123' in 's1.txt', '456' in 's2.txt', '789' in 's3.txt'
data_rdd = sc.parallelize(['s1.txt', 's2.txt', 's3.txt'])

result_rdd = data_rdd.map(myfun)

print(result_rdd.collect()) 
# output
# [(b'10.xx.xx.xx \n', '/home/xxx/tmp/spark-2.0.2-bin-hadoop2.7'), (b'xx.xx.xx.xx \n', '/home/xxx/tmp/spark-2.0.2-bin-hadoop2.7'), (b'xx.xx.xx.xx \n', '/home/xxx/tmp/spark-2.0.2-bin-hadoop2.7')]

print('driver program stopped')


spark.stop()
