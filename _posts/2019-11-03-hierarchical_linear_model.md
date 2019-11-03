---
layout: post
title: Hierarchical linear model
subtitle: Hierarchical linear model
category: Statistics
use_math: true
---

우선, Hierarchical Linear Model을 살펴보기 위하여 linear regression의 간단한 example부터 이야기를 시작해보도록 하자.  

### 1. Simple Linear Regression
한 학교에서 학생들의 경제학 점수를 $X_i$, 수학 점수를 $Y_i$라고 하자. 이 때 다음과 같은 회귀 모형을 만들 수 있다.

$$ Y_i = \beta_0 + \beta_1 X_i + \gamma_i$$

이 때 $\beta_0$는 경제학 점수가 0점인 학생의 expected 수학 점수다. $\beta_1$은 경제학 점수가 한 단위 증가할 때의 기대 수학 점수 변화이다. Error term인 $\gamma_i$는 i 번째 사람의 unique effect라고 할 수 있다. 일반적인 회귀 모형에서는 $\gamma_i$의 분포에 대하여 다음과 같이 가정한다.

$$ \gamma_i \sim N(0, \sigma^2)$$

<br>
<br>
### 2. Simple Linear Regression: Centering

이 때 intercept $\beta_0$가 meaningful하게 만들어 주기 위하여, 우리는 X를 'centering'을 통해 scaling해줄 수 있다.

$$ Y_i = \beta_{0*} + \beta_{1*} (X_i-\overline{X}) + \gamma_{i*}$$

즉 $X_i$대신 $X_i-\overline{X}$를 사용하는 것이다. 이 때 slope $\beta_{1*}$에 대한 해석은 바뀌지 않지만, intercept $\beta_{0*}$에 대한 해석이 바뀌게 된다. 'centering'후 $\beta_{0*}$는 '평균 수학 점수'로 해석할 수 있게 된다.

<br>
<br>
### 3. Two models

위와 같은 회귀 모형을 A학교와 B학교에 적용하여 각각 하나의 회귀 모형을 얻을 수 있다.  

<img src = '/post_img/191103/two_schools.png' width="250"/>
