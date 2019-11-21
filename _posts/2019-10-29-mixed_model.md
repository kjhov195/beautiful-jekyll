---
layout: post
title: Linear mixed model
subtitle: Linear mixed model
category: Statistics
use_math: true
---

<br>
<br>
### Mixed effect & Random effect

y의 공분산행렬은 실제로는 아주 복잡할 것이다. 하지만 model을 간단하게 만들기 위하여 공분산 term을 간단하게 나타내줄 필요가 있는데, 이 때 random effect를 가정하면 공분산 행렬이 간단한 형태를 가진다는 가정에 타당한 근거가 생기게 된다.

즉, Mixed effect는 $E[y]$를 모델링하고, random effect는 $Var[y]$를 모델링하는데 사용된다는 것이 핵심이다.

<br>
<br>
### Linear mixed model

우리가 일반적으로 Linear model이라고 칭하는 모델은 다음과 같은 fixed effect model이다. fixed effect model에서 $\beta$는 fixed이다.

$$
\begin{align*}
&Y = X\beta + e\\
&E[Y] = X\beta\\\\
&where\;
\begin{cases}
E(e) = 0\\
Var(e) = R\\
\end{cases}
\end{align*}
$$

반면, Mixed model의 경우 fixed effect 뿐만 아니라, random effect를 포함하는 모델이다. 다음의 mixed effect model에서 $\alpha$는 fixed effect, $\beta$는 random effect에 해당한다.

$$
\begin{align*}
&Y = X\beta+Zu+e,\\
&E[Y \vert u] = X\beta+Zu\\\\
&where
\begin{cases}
E(e) = 0\\
Var(e) = R\\
Var(Y|u) = R\\
E(u) = 0\\
Var(u) = D
\end{cases}
\\\\
\end{align*}
$$

이 때 $y$의 unconditional distribution은 다음과 같다.

$$
y \sim (X\beta, ZDZ'+R)
$$

linear mixed model에는 회귀분석, ANOVA, ANCOVA, hierarchical linear model 등 많은 모형이 속해있다. 이러한 모형들은 모두 위의 general한 linear mixed model form에서 $X, \beta, Z, D, R$ 만 적절하게 바꿔주면 된다.

<br>
<br>
### Random effect는 Var(y)를 모델링한다.

fixed effect는 y의 평균을 모델링하는데 사용되고, random effect는 y의 분산을 모델링하는데 사용된다.

y의 분산을 모델링한다는 말에 대하여 조금 더 자세히 살펴보도록 하자.

우선 다음과 같은 모델을 가정해보자.

<br>

$$ y_{tijk} = \beta_t + s_i + c_j + \epsilon_{tijk}$$

$$ E[y_{tijk} \vert s_i, c_j] = \beta_t + s_i + c_j $$

$$ \text{where }\beta_t \text{ is fixed effect},\\
\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;s_i, c_j \text{ is random effects}$$

<br>

우리는 $ E[y|u] = X\beta + ZU$의 형태로 모델링하고자 한다. 현재 우리가 모델에서 가정한 random effect는 $s_i$, $c_j$ 두 개이다. U와 Z를 다음과 같이 가정한다면 $ZU = z_1 u_1 + z_2 u_2$로 나타낼 수 있을 것이다.

<br>

$$
Z =
\begin{bmatrix}
Z1 & Z2
\end{bmatrix}\\
U =
\begin{bmatrix}
u1\\
u2
\end{bmatrix}
$$

<br>

이렇게 두게 된다면, random effect의 분산 D 또한 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
D &= Var(u) \\
&= Var(
\begin{bmatrix}
u1\\
u2
\end{bmatrix}
)\\
&=
\begin{bmatrix}
Var(u1) & Cov(u_1, u_2)\\
Cov(u2, u1) & Var(u2)
\end{bmatrix}\\
&=
\begin{bmatrix}
D_1 & D_{12}\\
D_{12} & D_2
\end{bmatrix}
\end{align*}
$$

결과적으로,

$$
\begin{align*}
Var(Y) &= ZDZ^T+R\\
&=
\begin{bmatrix}
Z1 & Z2
\end{bmatrix}
\begin{bmatrix}
D_1 & D_{12}\\
D_{12} & D_2
\end{bmatrix}
\begin{bmatrix}
Z1\\
Z2
\end{bmatrix}
+R\\
&= Z_1 D_1 Z_1^T + Z_2 D_2 Z_2^T + Z_1 D_{12} Z_2^T + Z_2 D_{21} Z_1^T + R\\
&= \sum_{i=1}^r Z_i D_{ii} Z_{i}^T + \sum_{i=1}^r \sum_{i'=1}^r Z_i D_{ii'} Z_{i'}^T + R \;\;\;\;\;\text{In this examle, r=2}
\end{align*}
$$

<br>

여기서 우리는 일반적으로 random effect들 간의 covariance를 0이라고 가정하게 된다.(이 가정은 크게 무리 없는 가정이다.)

$$ D_{12} = D_{21} = 0$$

<br>

그렇다면 다음과 같이 다시 $D$와 $ZDZ^T + R$ 을 나타낼 수 있다.
$$
\begin{align*}
D &= Var(u) \\
&=
\begin{bmatrix}
D_1 & 0\\
0 & D_2
\end{bmatrix}\\
\\
Var(Y) &= ZDZ^T+R\\
&=
\begin{bmatrix}
Z1 & Z2
\end{bmatrix}
\begin{bmatrix}
D_1 & 0\\
0 & D_2
\end{bmatrix}
\begin{bmatrix}
Z1\\
Z2
\end{bmatrix}
+R\\
&= Z_1 D_1 Z_1^T + Z_2 D_2 Z_2^T + R\\
&= \sum_{i=1}^r Z_i D_{ii} Z_{i}^T + R \;\;\;\;\;\text{In this examle, r=2}
\end{align*}
$$

Random effect를 두게 되면 위와 같이 y의 공분산행렬($Var(Y)$)이 간단한 형태를 가지게 된다.

따라서 간단하게 $Var(Y)$를 모델링하여 간단한 형태의 모델을 만들고자 할 때 Random effect를 모델에 넣어주면 간단한 공분산행렬에 대한 가정에 근거가 생기게 된다.




<br>
<br>
### Reference
Helen Brown, Robin Prescott (2015). Applied Mixed Models in Medicine. chapter 1.
