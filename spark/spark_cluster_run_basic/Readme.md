# 如何Run

* 部署一个Spark集群：1个Master，2个Slave（把程序执行需要用到的数据文件my_data.csv拷贝到2个Slave中）
* 在测试机（独立于集群）中运行如下命令，就把程序`myapp.py` submit到集群运行，集群的Master自动把任务分解到两个Slave上运行

```
./bin/spark-submit --master spark://host:ip myapp.py
```
