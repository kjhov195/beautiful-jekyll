---
layout: post
title: Streaming(2)-Structured Streaming
subtitle: Apache Spark
category: Spark
use_math: true
---

<br>
<br>
### Structured Streaming(구조적 스트리밍)

Structured Streaming은 Spark의 구조적 API를 기반으로 하는 고수준 스트리밍 API이다. 앞서 살펴본 Dstream API의 단점을 보완하여 더욱 발전시킨 것이 Structured Streaming이라고 생각하면 된다.

구조적 스트리밍의 기본적인 특징은 다음과 같다.

- 확장가능하고 내고장성(fault tolerance)을 가진 Spark SQL 엔진에 구축된 스트림 처리 엔진
- 스트리밍 연산은 정적 데이터에 대한 배치 연산을 표현하는 것과 비슷한 방식으로 표현 가능
- Spark SQL 엔진은 증분형으로 끊임없이 실행하며 스트리밍 데이터가 도착할 때마다 최종 결과를 업데이트
- Scala, Java, Python, R 에서 DataFrame/Dataset API를 사용하여 스트리밍 집계, 이벤트 시간 윈도우, 스트림-배치 조인 등을 표현 가능

Structured Streaming의 가장 기본이 되는 컨셉은 데이터 스트림을 '데이터가 연속적으로 추가되는 테이블'처럼 다루는 것이다.

<br>
<br>
### 구조

<br>

<center><img src = '/post_img/191115/image1.png' width="600"/></center>

[<center>Image: Microsoft, Structured streaming with Azure Databricks into Power BI & Cosmos DB</center>](http://microsoftonlineguide.blogspot.com/2018/06/structured-streaming-with-azure.html)

<br>

우리는 data stream을 Ubounded input table로 생각할 수 있다. 즉, input으로 데이터 스트림이 새로 추가될 때 마다 Input Table의 하나의 row로 들어가게 되며, 이러한 rows가 무한히 많이 추가될 수 있는 구조를 가지고 있다고 생각하면 된다.

<br>

<center><img src = '/post_img/191115/image2.png' width="600"/></center>

[<center>Image: databricks, Spark Structured Streaming</center>](https://databricks.com/blog/2016/07/28/structured-streaming-in-apache-spark.html)

<br>

개발자는 입력 테이블이 마치 정적 테이블인 것처럼 생각하여 출력 싱크에 작성될 최종 결과 테이블(result table)을 정의하는 쿼리를 정의한다.

Spark는 자동으로 이 쿼리를 스트리밍 실행 계획으로 전환하며, 이후에 레코드가 도착할 때마다 결과를 업데이트하는 데 필요한 상태를 파악하게 된다.

마지막으로 개발자는 트리거를 설정하여 언제 결과를 업데이트할지 제어한다. 트리거가 실행될 때마다 Spark는 새로운 데이터를 확인하고, 결과를 증분형으로 업데이트한다.

구조적 스트리밍은 소스로부터 스트림 데이터를 읽어들이고, 증분형으로 처리하여 결과를 업데이트한 후, 소스 데이터를 버린다. 결과 업데이트에 필요한 최소한의 중간 상태 데이터만 보관하는 것이다.

결과 테이블이 업데이트될 때마다 외부 시스템(예를들어 HDFS, DB)에 그 변화를 기록하고, 증분형으로 출력하게 된다.

<br>
<br>
### Input Source

Structured Streaming에는 소켓(socket), 파일, RDD 큐, Kafka와 같은 다양한 입력 소스를 사용할 수 있다.

<br>
#### 1. Socket

<br>
#### 2. File
<br>
#### 3. RDD Queue
<br>
#### 4. Kafka







writing...
수정 중...

<br>
<br>
### Reference
[YBIGTA Engineering Team](https://github.com/YBIGTA/EngineeringTeam)

YBIGTA Engineering Team 11기 김현우, SPARK Streaming

YBIGTA Engineering Team 13기 정우담, SPARK Streaming

YBIGTA Engineering Team 14기 한승희, Spark Streaming

<br>
<br>
