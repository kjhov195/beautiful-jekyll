---
layout: post
title: Spark install(2)-Java,Scala,Hadoop
subtitle: Java, Scala, Apache Hadoop
category: Spark
use_math: true
---

<br>
<br>
### 1. Java

```
$ cd $HOME
$ sudo apt-get install openjdk-8-jdk

$ nano ~/.bash_profile
```

다음을 추가해준다.

```
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# Anaconda
export ANACONDA_HOME="/home/kjhov195/anaconda3"

# Java
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Path
export PATH=${ANACONDA_HOME}/bin:${JAVA_HOME}/bin:$PATH
```

변경된 환경 변수를 적용해준다.

```
source ~/.bash_profile
```

<br>
<br>
### 2. Scala

```
sudo apt-get install scala
```

<br>
<br>
### 3. Hadoop(Pseudo-Dstributed mode)
<br>
#### 1. install protobuf
```
$ sudo apt-get install gcc
$ sudo apt-get install g++
$ sudo apt-get install make

$ cd /usr/local
$ sudo wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
$ sudo tar xvzf protobuf-2.6.1.tar.gz
$ cd protobuf-2.6.1

$ sudo ./configure
$ sudo make
$ sudo make install
$ sudo ldconfig

$ protoc --version
```

<br>
<br>
#### 2. install Hadoop

```
$ cd $HOME
$ wget https://archive.apache.org/dist/hadoop/core/hadoop-2.9.0/hadoop-2.9.0.tar.gz
$ tar xvzf hadoop-2.9.0.tar.gz

$ ln -s hadoop-2.9.0 hadoop

$ mv hadoop-2.9.0.tar.gz ./downloads/
```

<br>
<br>
#### 3. setting for hadoop
<br>
##### hadoop-env.sh
```
$ cd $HOME/hadoop/etc/hadoop
$ nano hadoop-env.sh
```

다음 두 부분을 찾아 수정해준다.

```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export HADOOP_PID_DIR=/home/kjhov195/hadoop/pids
```
<br>
##### masters
```
$ nano masters
```
다음의 내용으로 수정해준다.

```
local host
```
<br>
##### slaves
```
$ nano masters
```
다음의 내용으로 수정해준다.

```
local host
```
<br>
##### core-site.xml
```
$ nano core-site.xml
```
기존의 내용을 지우고 다음의 내용으로 수정해준다.

```
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9010</value>
  </property>
</configuration>
```
<br>
##### hdfs-site.xml
```
$ nano core-site.xml
```
기존의 내용을 지우고 다음의 내용으로 수정해준다.

```
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>/home/kjhov195/data/dfs/namenode</value>
  </property>
  <property>
    <name>dfs.namenode.checkpoint.dir</name>
    <value>/home/kjhov195/data/dfs/namesecondary</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/home/kjhov195/data/dfs/datanode</value>
  </property>
  <property>
    <name>dfs.http.address</name>
    <value>localhost:50070</value>
  </property>
  <property>
    <name>dfs.secondary.http.address</name>
    <value>localhost:50090</value>
  </property>
</configuration>
```
<br>
##### hdfs-site.xml

```
$ cp mapred-site.xml.template mapred-site.xml
$ nano mapred-site.xml
```

다음의 내용으로 수정해준다.

```
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>
```
<br>
##### yarn-site.xml
```
$ nano yarn-site.xml
```
기존의 내용을 지우고 다음의 내용으로 수정해준다.

```
<?xml version="1.0"?>

<configuration>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>localhost</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
  </property>
  <property>
    <name>yarn.nodemanager.local-dirs</name>
    <value>/home/kjhov195/data/yarn/nm-local-dir</value>
  </property>
  <property>
    <name>yarn.resourcemanager.fs.state-store.uri</name>
    <value>/home/kjhov195/data/yarn/system/rmstore</value>
  </property>
</configuration>
```
<br>
##### bash_profile

```
$ nano ~/.bash_profile
```

다음과 같이 수정해준다.

```
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# Anaconda
export ANACONDA_HOME="/home/kjhov195/anaconda3"

# Java
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Hadoop
export HADOOP_HOME="/home/kjhov195/hadoop"

# Path
export PATH=${ANACONDA_HOME}/bin:${JAVA_HOME}/bin:${HADOOP_HOME}/bin:$PATH
```

환경변수를 적용해준다.

```
$ source ~/.bash_profile
```


#### 4. ssh
```
$ cd $HOME
$ ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

$ ssh localhost

$ exit
```
local host에 접속 가능한 것을 확인할 수 있다.


#### 5. run hadoop
```
$ cd $HADOOP_HOME

$ ./bin/hdfs namenode -format

# 데몬 실행
$ ./sbin/start-dfs.sh
$ ./sbin/start-yarn.sh

# 실행된 프로세스 확인
$ jps
```

다음과 같은 결과가 보인다면 데몬이 정상적으로 실행된 것이다.

```
18469 NodeManager
18728 Jps
18074 SecondaryNameNode
17676 NameNode
18252 ResourceManager
17853 DataNode
```

```
# 데몬 중지
$ ./sbin/stop-yarn.sh
$ ./sbin/stop-dfs.sh
```

<br>
<br>
