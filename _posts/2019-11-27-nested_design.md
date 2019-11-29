---
layout: post
title: Nested Design
subtitle: Design of Experiment
category: Statistics
use_math: true
---

<br>
<br>
### 1. Nested Design

우선, Two-stage nested design을 생각해보자. 3명의 Suppliers(1,2,3)에게 물건을 공급받는다고 하자.

우리는 각각의 Supplier로부터 공급받은 물건의 품질이 같은지 궁금하여 실험하고 싶다. 이 때 각 공급자들로부터 공급받은 물건들을 각각 4개의 Batches로 나눌 수 있다.

<center><img src = '/post_img/191127/image1.png' width="600"/></center>

위 그림과 같이 나누어진 각각의 Batch들을 다음과 같이 1,2,3,4로 표현할 수 있다.

하지만 이 때 Supplier 1의 Batch 1과, Supplier 2의 Batch 1, Supplier 3의 Batch 1은 결코 같은 Batch가 아니며, 아무런 관계가 없다.

즉, 같은 숫자의 Batch라고 하여도 전혀 다른 Batch인 것이다. 그렇기 때문에 우리는 다음과 같이 Batch에 다른 Numbering을 하는 것 또한 가능하다.

<center><img src = '/post_img/191127/image2.png' width="600"/></center>

이렇게 Batch들에게 re-numbering을 할 수 있다면, __Nested Desgin__ 이며, _Factor is nested_ 라고 표현한다.


<br>
<br>
### 2. Example

다음과 같은 모델을 생각해보자.

$$y_{ijk} = \mu + \tau_i + \beta_{j(i)}+\epsilon_{(ij)k}$$

$$
\begin{align*}
where\;i &= 1,2,3,...,a\\
j &= 1,2,3,...,b\\
k &= 1,2,3,...,n
\end{align*}
$$

__Factor A__ 내의 각각의 level($i = 1,2,3,...,a$)들 __밑에 Factor B__ 의 각각의 level($j = 1,2,3,...,b$)이 __Nested__ 되어 있다고 하자.

이 때 $\beta_{j(i)}$는 Factor B의 $j$번째 level이 Factor A의 $i$번째 level 밑에 __nested__ 되어 있다는 뜻이다.


<br>
<br>
### Reference

Douglas C. Montgomery(2017), Design and Analysis of Experiments, 9th Edition

<br>
<br>
