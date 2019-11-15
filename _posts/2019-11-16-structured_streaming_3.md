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

다음 상황을 상상해보자.

정오(12:00 pm)부터 5분 간격으로 최근 10분 동안의 데이터를 집계하고자 한다. 12:00에는 11:50~12:00까지의 데이터를 처리하고, 12:05에는 11:55~12:05까지의 데이터를 처리하고자 하는 것이다. 이 경우, 어플리케이션이 종료되기 전까지 5분마다 같은 방식으로 데이터를 읽고 처리해야 한다. 이러한 방법을 Window operation이라고 한다.

Window는 '시간 간격'이다. Streaming시 윈도우라는 시간 간격을 정해놓고, 일정 시간마다 주기적으로 윈도우의 크기 만큼의 데이터를 읽어와서 처리하는 것을 윈도우 연산이라고 한다.

<br>

<center><img src = '/post_img/191116/image1.png' width="800"/></center>

[<center>Image: Apache Spark, Window operation</center>](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)


<br>

위 그림의 예시를 시간대 별로 살펴보자.

12:05 => (12:05 이전에 들어온) 12:02, 12:03 데이터가 (12:00-12:10) 윈도우에 update

12:10 => (12:10 이전에 들어온) 12:07 데이터가 (12:00-12:10) 윈도우, (12:05-12:15) 윈도우에 모두 update

위 예시에서 windowDuration은 Window의 크기(시간), 즉 처리하는 시간의 길이를 뜻하며 예시에서 windowDuration=10이다.

slideDuration은 window가 이동하는 간격(시간)을 뜻한다. 예시에서 slideDuration=5이다.

startTime은 첫 batch가 시작하는 시간을 뜻한다.

<br>
<br>
### 이벤트 유효 시간

스트리밍 처리에서 가장 중요한 두 시각은 "이벤트 발생 시간", "시스템에 의해 이벤트가 감지된 시간"이다.

이 두 시각은 네트워크 상황에 따라 차이가 발생할 수 있다.

<br>

<center><img src = '/post_img/191116/image2.png' width="800"/></center>

[<center>Image: Apache Spark, Window operation</center>](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)


<br>


예를 들어, 12:04분에 발생한 이벤트가 12:11분에 시스템에 의해 감지될 수도 있는 것이다.

그래서 Window를 계속 유지하면서 늦게 도착하는 데이터를 받아들이는 방법을 사용할 수도 있다. 하지만 메모리에 이전의 결과를 계속 쌓아 두어야 하므로 이는 좋지 않은 방법이다.

즉, 이러한 유효 시간의 문제는 Window operation만으로는 해결하기가 쉽지 않다.

<br>
<br>
### Watermarking

워터마킹은 이벤트 유효 기간을 설정하여 이러한 문제를 해결한다.

트리거가 실행될 때마다 유효한 이벤트를 판별할 수 있는 기준 시간을 정한다.

다음 트리거가 발생할 때까지는 해당 시간을 기준으로 이벤트의 유효성을 판별하여 유효하지 않은 이벤트의 경우 최종 결과에 반영하지 않고 버린다.

이를 워터마킹이라고 한다.

<br>

<center><img src = '/post_img/191116/image3.png' width="800"/></center>

[<center>Image: Apache Spark, Watermarking</center>](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

<br>

유효성을 판별하는 기준 시각은 해당 트리거가 발생하기 전에 들어온 모든 이벤트 중에서

(가장 마지막에 발생된 이벤트의 발생 시각) - (사용자가 미리 지정해 둔 유효 기간)

으로 계산한다.

예를 들어, 5분마다 트리거를 발생시킨다고 하자. 트리거 발생 시간과 가장 가까운 이벤트의 발생시간에서 지연시간을 뺀 시각이 마지노선이라고 생각할 수 있다. 이 때 마지노선을 넘지 않는 window의 값을 최종 결과에 추가한다.

예를 들어, 12:15의 trigger에서 가장 최근의 이벤트 발생시간은 12:14이고, 지연시간은 10분이므로 12:04가 마지노선이다.

<br>
<br>
### Reference
[YBIGTA Engineering Team](https://github.com/YBIGTA/EngineeringTeam)

[Spark The Definitive Guide(스파크 완벽 가이드)](http://www.hanbit.co.kr/store/books/look.php?p_code=B6709029941)

<br>
<br>
