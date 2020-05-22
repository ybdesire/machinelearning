# File distributed processing program

## Introduction

This is an simple distributed feature engineering demo.

**What is the task?**

* I have a files list, such as ['s1.txt', 's2.txt', 's3.txt']
* The file features should be extracted at each slave of spark cluster
* How can we complete the task?


## Pseudo code logic

* (1) File list to RDD.

```
data_rdd = sc.parallelize(['s1.txt', 's2.txt', 's3.txt'])
```

* (2) Write function to process each file. Consider the parameter as each item of the rdd list

```
def myfun(m):
    filename = m
	download_file(filename)
	fea = extract_file_feature(filename)
    return fea
```

* (3) Call the function by rdd.map(). And check the result by rdd.collect()

```
result_rdd = data_rdd.map(myfun)
print(result_rdd.collect())
```


## How to run the program


* (1) Build a spark cluster with 1 master and 6 slave.
   * Start master: `./sbin/start-master.sh -h master_ip`
   * Start slave: `./sbin/start-slave.sh spark://master_ip:7077`
* (2) Modify the ftp related information at `file_distributed_process.py`.
* (3) Submit the program `file_distributed_process.py` by cmd below.

```
./bin/spark-submit --driver-memory 8g --executor-memory 6G --master spark://master_ip:7077 file_distributed_process.py
```