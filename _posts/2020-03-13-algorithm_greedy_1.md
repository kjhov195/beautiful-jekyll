---
layout: post
title: Algorithm-체육복
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200313/image1.png' width="450"/></center>
<center><img src = '/post_img/200313/image2.png' width="450"/></center>

<br>
### 이해

알고리즘 문제를 풀때 언제나 그렇듯, 사람은 어떠한 방식으로 이러한 문제를 푸는가를 생각해보아야 한다. 사람은 통찰력이 있기 때문에, input의 크기가 크지 않다면 한눈에 보고도 이 문제를 풀 수 있다. 그러나 프로그램은 통찰이 아닌, 단계별로 하나하나 차근차근 문제를 풀어갈 수 있도록 알고리즘을 구성해주어야만 이 문제를 해결할 수 있다.

이 문제의 경우, 탐욕법(Greedy Algorithm)을 사용하여 해결할 수 있다.

<br>
### 탐욕법(Greedy Algorithm)

Greedy Algorithm은 __알고리즘의 각 단계에서 그 순간에 최적이라고 생각되는 것을 선택__ 하는 방법을 의미한다.

Greedy Algorithm으로 최적의 해를 찾을 수 있는 경우는 __현재의 선택이 마지막 해답의 최적성을 해치지 않을 때__ 이다.

<br>
### 풀이

이 문제의 경우, 빌려줄 학생들을 __정해진 순서__ 로 살펴야 하고, 이 __정해진 순서__ 에 따라 우선하여 __빌려줄 방향__ 을 정해야 한다.




<br>
### Reference

https://hsin.hr/coci/archive/2009_2010/contest6_tasks.pdf
