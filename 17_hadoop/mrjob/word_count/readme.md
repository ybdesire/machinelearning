# environment

1. setup hadoop environment from zero

```
sudo apt-get update
apt install default-jre
apt install default-jdk
wget http://ftp.cuhk.edu.hk/pub/packages/apache.org/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz


JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
CLASSPATH=
PATH=$JAVA_HOME/bin:$PATH
export JAVA_HOME CLASSPATH PATH

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
export HADOOP_HOME=/home/tmp/hadoop-2.7.7
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin

```

2. install mrjob

```
pip install mrjob
```


# how to run

1. test.txt

```
Newly obtained documents confirm that James Comey’s FBI was running a secret and corrupt counterintelligence operation against the Trump campaign in the summer of 2016 and repeatedly deceiving the Foreign Intelligence Surveillance Court (FISC) thereafter in order to wiretap a Trump campaign associate.

The disclosure was the result of a federal lawsuit and a year of litigation. Despite efforts by FBI Director Christopher Wray to obstruct, a federal court issued an order that forced the FBI and Department of Justice to produce the records known as “302 reports.”  They are a summary of interviews FBI agents conducted with Bruce Ohr, a top DOJ official.

These 302s show that the FBI and DOJ were warned repeatedly by Ohr that ex-British spy Christopher Steele was virulently biased against the target of their investigation, Trump.

EX-FBI DEPUTY DIRECTOR ANDREW MCCABE SUES DOJ OVER DISMISSAL

That bias tainted the credibility of the “dossier” Steele composed and upon which officials in the Obama administration relied when they officially launched their counterintelligence investigation on July 31, 2016.  The “dossier” was also the basis for the surveillance warrant against former Trump campaign adviser, Carter Page.

The FBI and DOJ ignored the warnings of bias and actively concealed it from the FISC. They never advised the judges that the information contained in the “dossier” was “unverified.”

They hid from the judges that it was all funded by the Clinton campaign and the Democratic National Committee (DNC).
```

2. run mrjob code

```
python  mrjobtest.py  test.txt
```

3. output

```
"2016"  2
"302"   1
"302s"  1
"31"    1
"a"     7
"actively"      1
"administration"        1
"advised"       1
"adviser"       1
"against"       3
"agents"        1
"all"   1
"also"  1
"an"    1
"and"   9
"andrew"        1
"are"   1
"as"    1
"associate"     1
"basis" 1
"bias"  2
"biased"        1
"british"       1
"bruce" 1
"by"    3
"campaign"      4
"carter"        1
"christopher"   2
"clinton"       1
"comey" 1
"committee"     1
"composed"      1
"concealed"     1
"conducted"     1
"confirm"       1
"contained"     1
"corrupt"       1
"counterintelligence"   2
"court" 2
"credibility"   1
"deceiving"     1
"democratic"    1
"department"    1
"deputy"        1
"despite"       1
"director"      2
"disclosure"    1
"dismissal"     1
"dnc"   1
"documents"     1
"doj"   4
"dossier"       3
"efforts"       1
"ex"    2
"fbi"   7
"federal"       2
"fisc"  2
"for"   1
"forced"        1
"foreign"       1
"former"        1
"from"  2
"funded"        1
"hid"   1
"ignored"       1
"in"    4
"information"   1
"intelligence"  1
"interviews"    1
"investigation" 2
"issued"        1
"it"    2
"james" 1
"judges"        2
"july"  1
"justice"       1
"known" 1
"launched"      1
"lawsuit"       1
"litigation"    1
"mccabe"        1
"national"      1
"never" 1
"newly" 1
"obama" 1
"obstruct"      1
"obtained"      1
"of"    8
"official"      1
"officially"    1
"officials"     1
"ohr"   2
"on"    1
"operation"     1
"order" 2
"over"  1
"page"  1
"produce"       1
"records"       1
"relied"        1
"repeatedly"    2
"reports"       1
"result"        1
"running"       1
"s"     1
"secret"        1
"show"  1
"spy"   1
"steele"        2
"sues"  1
"summary"       1
"summer"        1
"surveillance"  2
"tainted"       1
"target"        1
"that"  7
"the"   24
"their" 2
"thereafter"    1
"these" 1
"they"  4
"to"    3
"top"   1
"trump" 4
"unverified"    1
"upon"  1
"virulently"    1
"warned"        1
"warnings"      1
"warrant"       1
"was"   6
"were"  1
"when"  1
"which" 1
"wiretap"       1
"with"  1
"wray"  1
"year"  1
```

# understand the code process

https://mrjob.readthedocs.io/en/stable/guides/concepts.html
