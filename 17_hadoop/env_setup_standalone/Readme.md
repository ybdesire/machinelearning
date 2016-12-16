# Hadoop Environment Setup (Standalone Mode)

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

## 2. Hadoop

* (1) Download hadoop-2.7.3 from [here]( http://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-2.7.3/hadoop-2.7.3.tar.gz)

* (2) `tar` to current folder

* (3) make sure command `bin/hadoop version` successfully 

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# bin/hadoop version
Hadoop 2.7.3
Subversion https://git-wip-us.apache.org/repos/asf/hadoop.git -r baa91f7c6bc9cb92be5982de4719c1c8af91ccff
Compiled by root on 2016-08-18T01:41Z
Compiled with protoc 2.5.0
From source with checksum 2e4ce5f957ea4db193bce3734ff29ff4
This command was run using /home/bin_yin/tmp/hadoop-2.7.3/share/hadoop/common/hadoop-common-2.7.3.jar
```