---
layout: post
title: Robust Prior(Diffuse/Mixture/Hierarchical)
subtitle: Bayesian Statistics
category: Bayesian
use_math: true
---

<br>
<br>
### Robust Prior

우리는 Prior를 설정할 때 수학적으로 편리하면서도 추론에 미치는 영향을 최대한 작게 만들어주는, 즉 robust한 prior를 설정해줄 필요가 있다. 이러한 concept에서 생각해낸 prior가 바로 robust prior이다.

<br>
<br>
### 1. Diffuse prior

가장 간단한 방법으로는 Diffusing이 있다. Conjugate prior의 분산을 상당히 크게 만들어주어 diffuse해주는 방법으로 robust prior를 줄 수 있다.

예를 들어 $\theta \sim N(0,1000^2)$과 같은 prior를 주면 flat prior는 아니지만 flat prior와 유사한 형태의 prior를 줄 수 있게 된다.

<br>
<br>
### 2. Mixture prior

또 다른 방법으로 여러 Prior distribution의 weighted sum으로 새로운 prior를 주는 방법이 있다. 이를 Mixture prior라고 한다. Mixture prior를 사용한다면 훨씬 더 유연한 prior를 설정할 수 있다는 장점이 있다.

<br>
##### Example

Likelihood: $y \vert \theta \sim B(n, \theta)$

Prior: $\theta \sim \pi Beta(\alpha_1, \beta_1) +(1-\pi)Beta(\alpha_2, \beta_2)$

Posterior:

$$
\begin{align*}
p(\theta \vert y) &\propto p(y \vert \theta) p(\theta)\\
&\propto \binom{n}{y} \theta^y (1-\theta)^{n-y} \left \lbrack
\pi {\Gamma(\alpha_1+\beta_1) \over \Gamma(\alpha_1)\Gamma(\beta_1)}\theta^{\alpha_1-1}(1-\theta)^{\beta_1-1}
+(1-\pi) {\Gamma(\alpha_2+\beta_2) \over \Gamma(\alpha_2)\Gamma(\beta_2)}\theta^{\alpha_2-1}(1-\theta)^{\beta_2-1}
\right \rbrack\\
&\propto \pi \theta^{y+\alpha_1-1}(1-\theta)^{n-y+\beta_1-1}
+(1-\pi) \theta^{y+\alpha_2-1}(1-\theta)^{n-y+\beta_2-1}\\
&\sim \pi Beta(y+\alpha_1,n-y+\beta_1)+(1-\pi)Beta(y+\alpha-1,n-y+\beta_2)
\end{align*}
$$

<br>
이 경우, Mixture prior를 준 결과 posterior 또한 mixture distributino의 형태로 얻어지는 것을 확인할 수 있다.



<br>
<br>
### 3. Hierarchical model

Hierarchical model은 prior의 모수에 대하여 또 다른 prior(Hyper Prior)를 가정하는 모델이다.

다음은 모수 $\theta$가 Hyper parameter $\mu$를 가지고 있고, $\mu$의 prior 즉 Hyper prior가 p인 경우를 각 level에 따라 나타낸 것이다.

$$Level\;1: y \vert \theta \sim p(y \vert \theta)$$

$$Level\;2: \theta \vert y \sim p(\theta \vert \mu)$$

$$Level\;3: \mu \sim p(\mu)$$

여기서 기억해야 할 것은 Hyper parameter $\mu$가 data $y$로부터 정의되지 않으므로 다음의 식이 성립한다는 것이다.

$$ p(y \vert \theta, \mu) = p(y \vert \theta)$$

<br>

다음 예시를 통해 조금 더 자세히 이해해 보도록 하자.

<br>
##### Example

Likelihood: $y \vert \theta \sim B(n, \theta)$

Prior: $\theta \vert (\alpha, \beta)\sim Beta(\alpha, \beta)$

Hyperprior: $(\alpha, \beta) \sim Gamma(a_1, a_2)Gamma(b_1,b_2)$

Posterior:

$$
\begin{align*}
p(\theta, \alpha, \beta \vert y) &\propto p(y \vert \theta, \alpha, \beta) p(\theta, \alpha, \beta)\\
& = p(y \vert \theta) p(\theta \vert \alpha, \beta)p(\alpha, \beta)\\
&\propto \theta^y(1-\theta)^{n-y}\theta^{\alpha-1}(1-\theta)^{\beta-1}\alpha^{a_1-1}e^{-a_2\alpha}\beta^{b_1-1}e^{-b_2\beta}
\end{align*}
$$

<br>
<br>
