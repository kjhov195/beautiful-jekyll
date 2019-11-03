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
<br>
<br>
<br>
### 2. OLS(Ordinary Least Squared estimation)

사실 linear regression을 설명할 때 말했듯이, OLS(Ordinary Least Squared Estimation)에서는 error에 대한 분포 가정, 즉 y에 대한 분포 가정이 필요하지 않다. OLS를 찾기에 앞서, 우선 $X^TX$의 Rank에 대해 살펴 볼 필요가 있다.

$X$의 차원을 $n \times p$라고 하자.($X_{n \times p}$)

$Rank(X) = p$ 일 때, Full rank라고 하며 $(X^TX)^{-1} = (X^TX)^{-}$ 이다. 반면,
$Rank(X) < p$ 일 때에는 $(X^TX)^{-1}$가 존재하지 않으며, $GG^{-}G = G$를 만족하는 Generalized Inverse Matrix $G = (X^TX)^{-}$는 무수히 많이 존재한다.

우선, $Rank(X) = p$, 즉 $(X^TX)^{-1}$이 존재하는 경우, $\hat{\beta}_{LSE}$를 다음과 같이 구할 수 있다.
<br>


$$Q(\beta) = (Y-X\beta)^T(Y-X\beta)$$

$$
\begin{align}
\hat{\beta}_{LSE} &= argmin_{\beta}Q(\beta)\\
&= (X^T X)^{-1}X^TY
\end{align}
$$

이 때 이에 따른 $\hat{Y}$은 다음과 같다.

$$
\begin{align}
\hat{Y} &= X\hat{\beta}_{LSE}\\ &= X(X^T X)^{-1}X^TY
\end{align}
$$

<br>
만약 $Rank(X) < p$ 여서 $(X^TX)^{-1}$이 존재하지 않는 경우, $\hat{\beta}_{LSE}$는 Generalized Inverse Matrix($M^-$)를 사용하여 다음과 같이 구할 수 있다.

<br>
$$
\begin{align}
\hat{\beta}_{LSE} &= argmin_{\beta}Q(\beta)\\
&= (X^T X)^{-}X^TY\\
\end{align}
$$
$\hat{Y}$은 다음과 같다.
$$
\begin{align}
\hat{Y} &= X\hat{\beta}_{LSE}\\ &= X(X^T X)^{-}X^TY
\end{align}
$$
