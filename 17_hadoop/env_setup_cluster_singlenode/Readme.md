# Hadoop Single Node Cluster Environment Setup

* Environment: Linux Ubuntu Server 16.04 LTS


## 1. Install Java

* (1) Download jre-8u111-linux-x64.tar.gz from [here](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html)

* (2) `tar` to `/usr/java/`

* (3) `vim /etc/profile` and adding below

```
JAVA_HOME=/usr/java/jre1.8.0_111
CLASSPATH=.:$JAVA_HOME/lib.tools.jar
PATH=$JAVA_HOME/bin:$PATH
export JAVA_HOME CLASSPATH PATH
```

* (4) make sure command `java` successfully 


## 2. Adding a dedicated Hadoop system user

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# sudo addgroup hadoop
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# sudo adduser --ingroup hadoop hduser
```

## 3. SSH configuration

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# su - hduser
hduser@ubuntu:~$ ssh-keygen -t rsa -P ""
hduser@ubuntu:~$ cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys
```

Check if SSH configuration ok. (`hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3` will be changed to `hduser@ubuntu`)


```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ ssh localhost
hduser@ubuntu:~$ exit
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$
```


## 4. Download Hadoop

* (1) Download hadoop-2.7.3 from [here]( http://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-2.7.3/hadoop-2.7.3.tar.gz)

* (2) `tar` to current folder

## 5. Disable firewall

```
ufw disable
```

## 6. config below files at user `hduser`


* (1) etc/hadoop/hadoop-env.sh

```
export HADOOP_OPTS=-Djava.net.preferIPv4Stack=true
export JAVA_HOME=/usr/java/jre1.8.0_111
```


* (2) etc/hadoop/core-site.xml

```
<property>
  <name>hadoop.tmp.dir</name>
  <value>/data1</value>
</property>

<property>
  <name>fs.default.name</name>
  <value>hdfs://localhost:54310</value>
</property>
```

* (3) etc/hadoop/mapred-site.xml

```
<property>
  <name>mapred.job.tracker</name>
  <value>localhost:54311</value>
</property>
```

* (4) etc/hadoop/hdfs-site.xml

```
<property>
  <name>dfs.replication</name>
  <value>1</value>
</property>

```


## 7. Format namenode

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ bin/hadoop namenode -format
```

## 8. Run

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ sbin/start-all.sh
This script is Deprecated. Instead use start-dfs.sh and start-yarn.sh
Starting namenodes on [localhost]
localhost: starting namenode, logging to /home/bin_yin/tmp/hadoop-2.7.3/logs/hadoop-hduser-namenode-ubuntu.out
localhost: starting datanode, logging to /home/bin_yin/tmp/hadoop-2.7.3/logs/hadoop-hduser-datanode-ubuntu.out
Starting secondary namenodes [0.0.0.0]
0.0.0.0: starting secondarynamenode, logging to /home/bin_yin/tmp/hadoop-2.7.3/logs/hadoop-hduser-secondarynamenode-ubuntu.out
starting yarn daemons
resourcemanager running as process 22512. Stop it first.
localhost: nodemanager running as process 22642. Stop it first.
```

## 9. HDFS operation

* mkdir at hdfs

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ bin/hadoop dfs -mkdir /123
```

* ls at hdfs

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ bin/hadoop dfs -ls /
Found 1 items
drwxr-xr-x   - hduser supergroup          0 2016-12-17 17:00 /123
```

* copy from local to hdfs

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ bin/hadoop dfs -copyFromLocal README.txt /123
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ bin/hadoop dfs -ls /123
Found 1 items
-rw-r--r--   1 hduser supergroup       1366 2016-12-17 17:02 /123/README.txt
```

## 10. Access Hadoop UI

http://x.x.x.x:8088/cluster/nodes

## 11. Stop

```
hduser@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3$ sbin/stop-all.sh
```


# Reference

* http://blog.csdn.net/iam333/article/details/16357021
* http://www.michael-noll.com/tutorials/running-hadoop-on-ubuntu-linux-single-node-cluster/
* http://www.michael-noll.com/tutorials/running-hadoop-on-ubuntu-linux-multi-node-cluster/


