# How to run the program


* (1) Build a spark cluster with 1 master and 6 slave.
   * Start master: `./sbin/start-master.sh -h master_ip`
   * Start slave: `./sbin/start-slave.sh spark://master_ip:7077`
   
* (2) Modify the ftp related information at `get_running_slave_address.py`.

* (3) Submit the program `get_running_slave_address.py` by cmd below.


```
./bin/spark-submit --driver-memory 8g --executor-memory 6G --master spark://master_ip:7077 get_running_slave_address.py
```

