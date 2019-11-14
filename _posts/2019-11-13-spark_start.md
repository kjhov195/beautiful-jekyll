---
layout: post
title: PySpark
subtitle: Apache Spark
category: Spark
use_math: true
---

<br>
<br>
### PySpark
<br>
#### Hadoop metastore 실행


Hadoop과 Hive를 실행한 후, pyspark를 실행해야 한다.

먼저 Hadoop을 실행한다.
```
$ $HADOOP_HOME/sbin/start-dfs.sh
$ $HADOOP_HOME/sbin/start-yarn.sh
```

Hive를 실행한다.
```
$ cd $HOME/hive
$ ./bin/hive --service metastore &
$ jps
```

다음 7개의 패키지가 실행되고 있는 것을 확인한다.
```
23874 Jps
21123 NameNode
21717 ResourceManager
21993 NodeManager
21305 DataNode
23786 RunJar
21535 SecondaryNameNode
```

PySpark를 실행해준다.
```
$ cd $HOME
$ pyspark
```

jupyter notebook에서 __sc__ 를 입력하여 다음을 확인하여 잘 실행되었음을 알 수 있다.

```
SparkContext

Spark UI
Version
v2.3.4
Master
local[*]
AppName
PySparkShell
```

<br>
<br>
### Errors
<br>
#### 1. Jupter permission Errors

```
$ cd certs
$ ls -alh

$ rm -r ./*
```
yes, yes로 답한다.

```
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```
<br>
#### 2. Jupter permission Errors

ssh key를 다시 생성해준다.

```
$ cd $HOME
$ ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

$ ssh localhost

$ exit
```
<br>
#### 3. background에서 실행되고 있는 것을 중지

```
$ jobs
```

백그라운드에서 실행되고 있는 프로그램들의 목록이 뜬다.

여기서 번호를 찾아서, __kill -9 %숫자__ 로 중지할 수 있다.

예를들어, jobs로 확인하였을 때 1번 프로그램을 중지하고 싶다면 다음과 같이 입력한다.

```
$ kill -9 %1
```

<br>
<br>
