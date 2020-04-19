---
layout: post
title: Generalized Linear model(GLM)
subtitle: GLM
category: Statistics
use_math: true
---

<br>
<br>
### Generalized Linear Models

Generalized Linear Model은 Linear Model이 다음 두 가지 측면에서 확장되어 일반화 되는 것을 말한다.

<br>
<br>
### Exponential family

첫 번째는 $y$의 분포에 대한 것이다. Linear Model에서는 $y_i$들이 연속형 변수이며, 정규분포를 따른다고 가정하였다.

$$ y \sim N_N (X \beta, \sigma^2 I)$$

반면, GLM(Generalized Linear Model)에서는 $y_i$들이 exponential family를 따른다고 가정한다.

$$y_i \sim f_{Y_i}(y_i) = exp \left \lbrack {\frac {y_i\gamma_i-b(\gamma_i)} {\tau^2}} - c(y_i, \tau) \right \rbrack$$

이론적으로는 exponential family에 속하는 분포는 무한하게 많지만, 현실적으로 Normal, Binomial(Bernoulli), Poisson 분포가 GLM 응용 분야의 상당 부분을 차지한다고 봐도 무방하다.

<br>
<br>
### Link function

두 번째는 Link function에 대한 이야기이다. 우선 Link function이 무엇인지에 대하여 살펴볼 필요가 있다. 우리는 $E[y_i] = \mu_i$일 때, 다음을 가정할 수 있다.

$$ g(\mu_i) = x_i' \beta$$

이때 이 함수 $g(\cdot)$를 __link function__ 이라고 부른다.

Linear model에서는 다음과 같이 link function이 Identity Function이었다.

$$
\begin{align*}
E[y_i] &= \mu_i\\
g(\mu_i) &= \mu_i = x_i' \beta
\end{align*}
$$

반면 Generalized Linear Model에서는 link function이 Identity Function이 아닌 함수이다.

<br>
<br>
### Example 1. Logistic Regression

GLM의 대표적인 예시인 Logistic regression의 경우 link function은 다음과 같은 형태를 가진다.

$$
\begin{align*}
E[y_i] &= \mu_i\\
g(\mu_i) &= logit(\mu_i) = \log ({\mu_i \over 1-\mu_i}) = x_i' \beta
\end{align*}
$$

이를 $\mu_i$에 대해 다시 정리하면 우리가 잘 알고있는 로지스틱 함수가 된다.

$$ \mu_i = logistic(x_i' \beta) = \frac 1 {1+e^{-x_i' \beta}} $$

참고로 logistic function은 logit function과 역함수의 관계에 있다.

$$logit(x) = log(odds) = log({x \over 1-x})$$

$$logit^{-1}(x) = logistic(x) = {1 \over 1+exp(-x)} = {exp(x) \over exp(x)+1}$$

<br>
<br>
### Example 2. Poisson Regression

또 다른 GLM의 대표적인 예시인 Poisson regression의 경우 link function은 다음과 같은 형태를 가진다.

$$
\begin{align*}
E[y_i] &= \mu_i\\
g(\mu_i) &= log(\mu_i) = x_i' \beta
\end{align*}
$$

<br>
<br>
### Reference
Seung-Ho Kang(2019), Generalized Linear Model : Generalized Linear models, Yonsei University, p.1.
