# !/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel

spark = SparkSession\
        .builder\
        .appName("Kmeans at Cluster")\
        .getOrCreate()


# create spark context, which can be used to create RDD
sc = spark.sparkContext


# data
data = sc.parallelize([[1,1], [0,1], [1,0], [0,0], [12,21], [11, 21], [10, 21], [12, 20]])


# kmeans
model = KMeans.train(data, 2)
print("Final centers: " + str(model.clusterCenters)) # Final centers: [array([ 11.25,  20.75]), array([ 0.5,  0.5])]
print("Total Cost: " + str(model.computeCost(data))) # Total Cost: 5.499999999999545
print("Result: {0}".format(model.predict(data).collect())) # Result: [1, 1, 1, 1, 0, 0, 0, 0]


spark.stop()
