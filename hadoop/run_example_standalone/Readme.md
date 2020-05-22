# Run Hadoop self-example at Standalone mode

## 1. Setup Hadoop Standalone environment as [here](https://github.com/ybdesire/machinelearning/blob/master/17_hadoop/env_setup_standalone).

## 2. Environment setup successfully if command `bin/hadoop version` successfully.

## 3. Create folder and copy files for program running preparation.

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# mkdir my_test/input
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# cp *.txt my_test/input/
```

## 4. Run Hadoop example as below

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.3.jar wordcount my_test/input/ my_test/output
```

## 5. Check result

```
(py35project) root@ubuntu:/home/bin_yin/tmp/hadoop-2.7.3# cat my_test/output/part-r-00000
```

And we can get the word count as below.

```
with    103
withdraw        2
within  20
without 42
work    11
work,   3
work.   2
works   4
works,  1
world-wide,     4
worldwide,      4
```