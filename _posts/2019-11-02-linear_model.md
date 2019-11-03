---
layout: post
title: Linear Model
subtitle: Linear Model
use_math: true
---

Linear Model은 Linear Regression(회귀분석), ANOVA(분산분석), ANCOVA(공분산분석)을 모두 포함하는 모델이다. 전통적인 Linear Model에서는 다음을 가정한다.

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

<br>
<br>
### 3. MLE(Maximum Likelihood Estimation)
MLE를 구하기 위해서는 Normality에 대한 가정이 필요하다. 이제 $y$에 대하여 다음과 같이 다변량 정규분포를 가정해보자.

$$ Y \sim N_{N}(X\beta, \sigma^2 I_N)$$

Likelihood는 다음과 같이 구할 수 있다.

$$
\begin{align}
L = L(\beta, \sigma^2 | y) = (2\pi)^{-{1\over2}N} \exp[-{1\over2}(Y-X\beta)^T {1\over{\sigma^2}} (Y-X\beta)]
\end{align}
$$

log likelihood를 구하고,

$$
\begin{align}
l = log(L) = -{1\over2}N \log(2\pi) - {1\over2}N \log(\sigma^2) -{1\over2}(Y-X\beta)^T {1\over{\sigma^2}} (Y-X\beta)
\end{align}
$$

$l$을 미분하여 MLE를 찾아보면,

$$
\begin{align}
\\
\frac{\partial l}{\partial \beta} &= {1 \over {\sigma^2}}(2X^TY-2X^TX\beta) \overset{let}{=} 0\\\\
\hat\beta_{mle} &=
\begin{cases}
(X^T X)^{-1}X^TY \;\;\; if\;\;(X^TX)^{-1}\;exists\\
\\
(X^T X)^{-}X^TY \;\;\;\; else
\end{cases}
\end{align}
$$

<br>
<br>
#### c.f. $(X^TX)^{-}$ : Not Unique
$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; : (X^TX)^{-}$가 무수히 많이 존재할 수 있는데 괜찮을까?

<br>
이 때 $(X^TX)^{-1}$이 존재하지 않는 경우, Generalized Inverse Matrix $(X^TX)^{-}$는 무수히 많이 존재하여 $\hat{\beta}_{mle}$ 를 Unique하게 구할 수는 없다.
<br>
<br>
하지만  많은 경우 우리는 $\hat{\beta}$ 자체 보다는, $\hat E[Y] = \hat \mu = X\hat{\beta} $ 에 더 관심을 가지고 있으며, $\hat \mu$ 은 Unique하게 구할 수 있다. 즉, $\beta$를 Unique하게 추정할 수는 없지만, $E[Y]$는 Unique하게 추정할 수 있는 것이다.

<br>
<br>
$$
\begin{align}
\hat{\beta} &: not\;unique\\\\
\hat{\mu} &= X\hat{\beta}\\ &= X(X^T X)^{-}X^TY \;\;: unique\\
(&\because\; X(X^TX)^{-}X^T\;is\;invariant\;to\;(X^TX)^{-})
\end{align}
$$

<br>
<br>
이 때 $X\hat\beta$의 평균과 분산을 구해보면 다음과 같다.

$$
\begin{align}
E[\hat\mu] = E[X\hat\beta] &= X(X^TX)^{-}X^T E[Y]\\
&= X(X^TX)^{-}X^T X\beta \\
&= X\beta \;\;\;(\because X(X^TX)^{-}X^T X = X)\\
&= \mu\;\;\;\;\;: unbiased
\\
\\
Var[\hat\mu] = Var[X\hat\beta] &= Var[X(X^T X)^{-}X^TY]\\
&= X(X^T X)^{-}X^T \sigma^2 I X((X^T X)^{-})^TX^T\\
&= X(X^T X)^{-}X^T \sigma^2\;\;\;(\because X(X^TX)^{-}X^T X = X)
\end{align}
$$
