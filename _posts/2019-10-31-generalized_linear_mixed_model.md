---
layout: post
title: Generalized Linear Mixed model(GLMM)
subtitle: GLMM
category: Statistics
use_math: true
---

<br>
<br>
### Genealized Linear Mixed Model

GLMM(Genealized Linear Mixed Model)은 GLM(__Generalized__ Linear Model)과 LMM(Linear __Mixed__ Model)을 결합한 형태의 모형이다.

즉, Linear Model에 random effect를 추가하여 Linear Mixed model을 만들었던 것 처럼, Generalized Linear Model에 random effect를 추가하면 Generalized Linear Mixed Model을 만들 수 있다.

이전의 포스트에서 설명하였지만, random effect를 추가하게 되는 이유는 크게 두 가지가 존재한다.

첫 번째로 random effect를 추가함으로써 공분산행렬의 형태를 복잡하게 modeling하는 것이 가능해지고, 이러한 성질을 활용하여 Correlated 되어있는 데이터나, 반복적으로 측정된 자료를 다루는 모델을 만들 수 있기 때문이다.

두 번재로 전체 모집단에서 랜덤하게 뽑혀서 그 값이 결정된 factor를 모형에 반영하기 위함이다.


<br>
<br>
### Structure

GLMM의 일반적인 구조는 다음과 같이 구성할 수 있다.

첫 번재로 random effect $u$가 given일 때의 conditional distribution of $y$는 독립이라고 가정한다.

$$y_i \vert u \sim indep\;f_{Y \vert u}(y_i \vert u)$$

두 번재로 이렇게 정의한 Conditional distribution of $y$ given $u$가 Exponential family를 따른다고 가정한다.

$$
\begin{align*}
y_i \vert u &\sim indep\;f_{Y \vert u}(y_i \vert u)\\
&= exp \left \lbrack {y_i \gamma_i - b(\gamma_i) \over \tau^2 }- c(y_i, \tau)\right \rbrack
\end{align*}
$$

세 번째로 conditional mean of $y$ given $u$에 대한 어떠한 transformation $g(\cdot)$이 fixed effect와 random effect의 선형 모형이라고 가정한다.(여기서 $g(\cdot)$은 link function이다.)

$$
\begin{align*}
E[Y_i \vert u] &= \mu_i\\
g(\mu_i) &= x_i' \beta + z_i' u
\end{align*}
$$

네 번재로 random effect $u$의 분포를 가정한다.

$$u \sim f_U(u)$$

<br>
<br>
### Reference
Seung-Ho Kang(2019), Generalized Linear Model : Generalized Linear Mixed Models(GLMMs), Yonsei University, p.1.
