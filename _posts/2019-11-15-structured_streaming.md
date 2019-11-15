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
- Scala, Java, Python, R 에서 DataFrame /Dataset API 를 사용하여 스트리밍 집계, 이벤트 시간 윈도우, 스트림 배치 조인 등을 표현 가능

Structured Streaming의 가장 기본이 되는 컨셉은 '데이터 스트림을 데이터가 연속적으로 추가되는 테이블'처럼 다루는 것이다.

<br>
<br>
### 구조

<br>

<center><img src = '/post_img/191115/image1.png' width="600"/></center>

[<center>Image: Spark Structured Streaming</center>](https://databricks.com/blog/2016/07/28/structured-streaming-in-apache-spark.html)

<br>

writing...
수정 중...

<br>
<br>
### Reference
[YBIGTA Engineering Team](https://github.com/YBIGTA/EngineeringTeam)

YBIGTA Engineering Team 13기 정우담, SPARK Streaming

YBIGTA Engineering Team 14기 한승희, Spark Streaming

<br>
<br>
