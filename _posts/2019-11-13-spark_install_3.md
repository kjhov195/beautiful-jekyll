---
layout: post
title: Spark install(3)-Spark,Hive
subtitle: Apache Spark, Apach Hive
category: Spark
use_math: true
---

<br>
<br>
### 1. Apache Spark
<br>
#### py4j

```
$ conda install pip
$ which pip

$ pip install py4j
```
<br>
#### Spark

```
$ cd $HOME

$ wget http://apache.mirror.cdnetworks.com/spark/spark-2.3.4/spark-2.3.4-bin-hadoop2.7.tgz
$ tar xvzf spark-2.3.4-bin-hadoop2.7.tgz
$ ln -s spark-2.3.4-bin-hadoop2.7 spark
$ mv spark-2.3.4-bin-hadoop2.7.tgz ./downloads/

$ cd $HOME/spark/conf
$ cp spark-env.sh.template spark-env.sh
$ nano spark-env.sh
```

다음의 내용을 아래에 덧붙여줍니다.

```
export SPARK_MASTER_WEBUI_PORT=9090
export SPARK_WORKER_WEBUI_PORT=9091
export HADOOP_CONF_DIR=/home/kjhov195/hadoop/etc/hadoop
```

```
$ nano ~/.bash_profile
```

다음과 같이 수정해줍니다.

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

# Spark
export SPARK_HOME="/home/kjhov195/spark"

# PySpark
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'

# Path
export PATH=${ANACONDA_HOME}/bin:${JAVA_HOME}/bin:${HADOOP_HOME}/bin:${SPARK_HOME}/bin:$PATH
```

환경변수를 적용해준다.

```
$ source ~/.bash_profile
```

<br>
<br>
### 2. Apache Hive

```
$ cd $HOME

$ wget https://archive.apache.org/dist/hive/hive-2.3.4/apache-hive-2.3.4-bin.tar.gz
$ tar xvzf apache-hive-2.3.4-bin.tar.gz

$ ln -s apache-hive-2.3.4-bin hive

$ mv apache-hive-2.3.4-bin.tar.gz ./downloads/

$ cd $HOME
$ cd hive/conf
$ cp hive-env.sh.template hive-env.sh
$ nano hive-env.sh
```

다음 부분이 주석처리가 되어있는데, 해당 부분을 찾아서 다음과 같이 수정해준다.

```
HADOOP_HOME=/home/kjhov195/hadoop
```

다음 파일을 만들어서

```
$ nano hive-site.xml
```

다음과 같이 수정해준다.

```
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse/</value>
  </property>
  <property>
    <name>hive.cli.print.header</name>
    <value>true</value>
  </property>
  <property>
      <name>hive.metastore.uris</name>
      <value>thrift://localhost:9083</value>
  </property>
  <property>
      <name>hive.metastore.schema.verification</name>
      <value>false</value>
  </property>
</configuration>
```

hive를 실행하기 전에, Hadoop을 먼저 실행해준다.

```
# Hadoop 실행
$ $HADOOP_HOME/sbin/start-dfs.sh
$ $HADOOP_HOME/sbin/start-yarn.sh

# HDFS 디렉토리 생성/권한 부여
$ hdfs dfs -mkdir -p /tmp/hive
$ hdfs dfs -mkdir -p /user/hive/warehouse
$ hdfs dfs -chmod g+w /tmp
$ hdfs dfs -chmod 777 /tmp/hive
$ hdfs dfs -chmod g+w /user/hive
$ hdfs dfs -chmod g+w /user/hive/warehouse

# hive 메타스토어 초기화
$ cd $HOME/hive
$ ./bin/schematool -initSchema -dbType derby

# Hive 설정 파일을 spark/conf로 복사
$ cp $HOME/hive/conf/hive-site.xml $HOME/spark/conf/

$ ./bin/hive --service metastore &
```

jps를 입력하였을 때 다음과 같다면, 정상적으로 실행된 것이다.

```
21123 NameNode
21717 ResourceManager
21993 NodeManager
22745 RunJar
21305 DataNode
22830 Jps
21535 SecondaryNameNode
```
<br>
<br>
