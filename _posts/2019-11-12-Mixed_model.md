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
E(\beta) = 0\\
Var(\beta) = D
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
### Reference
Helen Brown, Robin Prescott (2015). Applied Mixed Models in Medicine. chapter 1.
