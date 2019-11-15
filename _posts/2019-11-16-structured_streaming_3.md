---
layout: post
title: Streaming 4 -Structured Streaming(Window, Watermarking)
subtitle: Apache Spark
category: Spark
use_math: true
---

<br>
<br>
### Window operation

Window는 '시간 간격'이다. Streaming시 윈도우라는 시간 간격을 정해놓고, 일정 시간마다 주기적으로 윈도우의 크기 만큼의 데이터를 읽어와서 처리하는 것을 윈도우 연산이라고 한다.

<br>

<center><img src = '/post_img/191116/image1.png' width="800"/></center>

_<center> [Image: Apache Spark, Window operation](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
 </center>_

<br>

예를들어 다음 상황을 상상해보자.

정오(12:00 pm)부터 5분 간격으로 최근 10분 동안의 데이터를 집계하고자 한다. 12:05에는 11:50~12:00까지의 데이터를 처리하며, 12:05에는 11:55~12:05까지의 데이터를 처리하고자 하는 것이다. 이 경우, 어플리케이션이 종료되기 전까지 5분마다 같은 방식으로 데이터를 읽고 처리해야 한다. 이러한 케이스에 Window operation을 사용할 수 있다.

windowDuration은 Window의 크기(시간), 즉 처리하는 시간의 길이를 뜻하며 예시에서 windowDuration=10이다.

slideDuration은 window가 이동하는 간격(시간)을 뜻한다. 예시에서 slideDuration=5이다.

startTime은 첫 batch가 시작하는 시간을 뜻한다.

<br>
<br>
###



작성 중

<br>
<br>
### Watermark operation

작성 중

<br>
<br>
### Reference
[YBIGTA Engineering Team](https://github.com/YBIGTA/EngineeringTeam)

<br>
<br>
