# Spark运行环境概述

* Java环境：Spark是用Scala写的，Scala最终会被编译为Java字节码，所以需要JVM环境
* Spark只需要下载gz文件，解压即可，不需要安装


# 搭建Spark运行环境（Linux CentOS）

* 安装jre：（下载和安装参考[这里](https://docs.oracle.com/javase/8/docs/technotes/guides/install/linux_jdk.html#BJFJJEFG)）
* 安装Python
* 关掉防火墙：`service iptables stop`，`chkconfig iptables off`


# 搭建Spark集群环境

集群环境是基于Master-Slave结构的

* 在所有Master-Slave节点上，下载Spark的gz文件，解压
* 在Master节点上，执行`./sbin/start-master.sh`
* 用Web浏览器连接Master的UI，看URL（spark://host:7077）
* 在所有Slave节点上，执行`./sbin/start-slave.sh spark://host:7077`
* Slave连接到Master，集群环境就搭建完成了


# 两个Python写Spark程序的入门例子

* [Python写Spark程序，在Spark单机上运行](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_local_run_basic)
* [Python写Spark程序，在Spark集群上运行](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_cluster_run_basic)


# Spark分布式程序

* [6 Slave 分布式特征提取例程](https://github.com/ybdesire/machinelearning/blob/master/16_spark/file_process_distributed)