---
layout: post
title: Missing data(2)-Statistical model
subtitle: Missing data
category: Statistics
use_math: true
---

<br>
<br>

앞서 Missing data(1) 포스트에서 고혈압 환자의 혈압 측정 예시를 통하여 Missing data에 대한 직관적인 이해를 중심으로 살펴보았다. 이번 포스트에서는 수식을 통하여 조금 더 구체적으로 Missing data와 modeling에 대하여 살펴보도록 하겠다.

<br>
<br>
### 0. Notation

$$
\begin{align*}
Y &= (Y_o, Y_m)
,\;\;\;where \;\begin{cases}
Y_o &: \text{observed values}\\
Y_m &: \text{missing values}\\
\end{cases}
\end{align*}
$$

y 값은 관측된($Y_o$:observed) y값과 관측되지 않은 y값($Y_m$:missing)으로 나눌 수 있다. 현실에서는 관측되지 않은 $Y_m$의 값은 알 수 없다.(관측되지 않았으므로 당연하다.)

우리가 알 수 있는 것은 $Y_m$이 아닌, 결측되었는지에 대한 여부 뿐이다. 따라서 다음과 같은 M matrix를 정의하여 결측 데이터를 나타낸다.

$$
\begin{align*}
Y &= (y_{ij})\\
M &= (m_{ij})\;\;where\;i=1,2,\cdots,n,\;j=1,2,\cdots,k \\
m_{ij} &= \begin{cases}
1\;\;\text{ if $y_{ij}$ is missing}\\
0\;\;\text{ if $y_{ij}$ is observed}
\end{cases}
\end{align*}
$$

when __MCAR__,

$f(M \vert Y_o, Y_m, \psi) = f(M \vert \psi)$

when __MAR__,

$f(M \vert Y_o, Y_m, \psi) = f(M \vert Y_o, \psi)$

when __MNAR__,

$f(M \vert Y_o, Y_m, \psi) = f(M \vert Y_o, Y_m, \psi)$


<br>
<br>
### 1. PDF

##### pdf of $Y$ (joint pdf of $Y_o, Y_m$)

$$f(Y \vert \theta) = f(Y_0, Y_m \vert \theta)$$

##### joint pdf of $Y, M$

$$f(Y,M \vert \theta, \psi) = f(Y \vert \theta) f(M \vert \psi)$$

##### joint pdf of $Y_o, M$

Full model에는 관측된 정보 뿐만 아니라, 관측되지 않은 정보 또한 포함되어야 한다.

따라서 Full model은 다음과 같다.

$$f(Y_o,M \vert \theta, \psi) = f(Y_o \vert \theta) f(M \vert \psi)$$

<br>
<br>
### 2. Likelihood

#### 1. Ignorable likelihood

$$L_{ign}(\theta \vert Y_o) = \prod_{i=1}^n f(Y_o \vert \theta)$$

$$f(Y_o \vert \theta) = \int f(Y_o, Y_m \vert \theta) dY_m$$


<br>
#### 2. Full likelihood

$$L_{full}(\theta, \psi \vert Y_o, M) = \prod_{i=1}^n f(Y_o,M \vert \theta, \psi)$$

$$f(Y_o,M \vert \theta, \psi) = \int f(Y_o, Y_m \vert \theta) f(M \vert Y_o, Y_m, \psi)dY_m$$

<br>
<br>
### 3. Full Likelihood under MAR

다음의 두 조건이 만족된다고 생각하자.

1. 결측치는 MAR

2. $(\theta, \psi)$의 joint parameter space가 각각의 parameter space의 product이다.

이 때 Full likelihood는 다음과 같이 정리할 수 있다.

$$L_{full}(\theta, \psi \vert Y_o, M) = \prod_{i=1}^n f(Y_o,M \vert \theta, \psi)$$

$$
\begin{align*}
f(Y_o,M \vert \theta, \psi) &= \int f(Y_o, Y_m \vert \theta) f(M \vert Y_o, Y_m, \psi)dY_m\\
&= \int f(Y_o, Y_m \vert \theta) f(M \vert Y_o, \psi)dY_m\;\;\;(\because MAR)\\
&= f(M \vert Y_o, \psi) \int f(Y_o, Y_m \vert \theta) dY_m\\
&= f(M \vert Y_o, \psi) f(Y_o \vert \theta)
\end{align*}
$$

<br>

여기서 알 수 있는 것은 MAR을 가정할 경우,

위 식에서 좌변 __$F(Y_o, M \vert \theta, \psi)$__ 를 최대화 시키는 것은 우변의 __$f(Y_o \vert \theta)$__ 를 최대화 시키는 것과 같다.

즉, __$L_{full}$__ 를 최대화 시키는 것은 __$L_{ign}$__ 를 최대화 시키는 것과 동일하다.

정리하면, Full liklihood를 최대화시킬 때 $M$을 고려할 필요 없이, 관측된 값의 joint pdf인 $Y_o$의 pdf $f(Y_o \vert \theta)$만 최대화시켜주면 되는 것이다.


<br>
<br>
### Reference
강승호(2019), 신약개발에 필요한 임상통계학, 자유아카데미


<br>
<br>
