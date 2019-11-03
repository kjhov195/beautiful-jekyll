---
layout: post
title: Linear Model
subtitle: Linear Model
use_math: true
---

# Linear Model

### Linear Models

Linear Models은 Linear Regression(회귀분석), ANOVA(분산분석), ANCOVA(공분산분석)을 모두 포함하는 모델이다. 전통적인 Linear Model에서는 다음을 가정한다.

### 1. Model

우선 error의 분포를 다음과 같이 가정해보자.

$$
\begin{align}
E(\epsilon) &= 0\\
Cov(\epsilon) &= \sigma^2 I_n\\
\epsilon &\sim N(0, \sigma^2 I_n)
\end{align}
$$

<br>
이 때 우리는 Response의 분포를 다음과 같다는 사실을 알 수 있다.

<br>
$$Y \sim (\mu, V)$$

<br>

$$
\begin{align}
where \;\;
\mu &=
\begin{bmatrix}
\mu_1 \\
\vdots\\
\mu_n
\end{bmatrix} =
\begin{bmatrix}
X_{11} & \cdots & X_{1p} \\
\vdots & \ddots & \vdots \\
X_{n1} & \cdots & X_{np}
\end{bmatrix}
\begin{bmatrix}
\beta_1 \\
\vdots\\
\beta_p
\end{bmatrix}
= X\beta\\
V &=
\begin{bmatrix}
\sigma^2 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \sigma^2
\end{bmatrix}
= \sigma^2 I_n
\end{align}
$$

<br>
즉, 결국 우리는 다음의 linear model을 가정한다.
<br>
<br>

$$ Y \sim (X\beta,\sigma^2 I_n)$$

<br>
여기서 Design Matrix $X_{n \times p}$는 Known이며, 모수들로 이루어진 $\beta$ 벡터는 unknown이다.
