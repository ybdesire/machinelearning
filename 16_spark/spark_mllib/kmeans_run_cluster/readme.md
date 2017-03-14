# Steps to run this program

* (1) one master, 6 slaves

* (2) start master by cmd below

```
./sbin/start-master.sh -h master-ip
```

* (3) submit my_kmeans.py

```
./bin/spark-submit --master spark://master-ip:7077 my_kmeans.py
```

* (4) get result

