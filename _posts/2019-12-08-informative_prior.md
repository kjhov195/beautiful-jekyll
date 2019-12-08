---
layout: post
title: Informative Prior
subtitle: Bayesian Statistics
category: Bayesian
use_math: true
---

<br>
<br>

Prior에 대해 우리는 많은 선택을 할 수 있다. Prior를 어떻게 선택하느냐는 결과와 해석에 아주 큰 영향을 미칠 수 있는데, 특히 small samples case의 경우 그 영향은 아주 커진다. 즉, 같은 데이터 임에도 불구하고 prior의 선택에 따라 다른 결론으로 이어질 수도 있는 것이다. 따라서 적절한 prior를 선택하는 것은 아주 중요한 issue가 될 수 있다.

Prior의 선택에 대해서는 크게 Informative/Non-informative Prior로 나누어 살펴볼 계획이다. 이번 post에서는 Informative Prior에 대해 살펴보도록 하겠다.

Prior를 선택할 때에는 편리성(Convenience), 연구자의 Intuition 등을 고려하여 선택하게 된다. 이 중 편리성 측면에서 살펴보면 크게 Conjugacy와 Vagueness를 중심으로 살펴볼 수 있다.

<br>
<br>
### Conjugate Prior

Informative prior의 대표적인 예로 Informative prior가 있다. Conjugacy는 켤레성이라는 뜻이다. Conjugate Prior는 특정 Likelihood와 Prior로 구한 Posterior의 분포가 해당 Prior의 분포와 같게 나오도록 하는 Prior를 말한다. 대표적으로 다음과 같은 Conjugate prior가 있다.

- Binomial likelihood + Beta prior = Beta Posterior

- Poisson likelihood + Gamma prior = Gamma Posterior

- Normal likelihood + Inv-Gamma Prior = Inv-Gamma Posterior

다음 예시를 통해 조금 더 자세히 살펴보도록 하자.

<br>
<br>
### Example 1- Conjugate prior

Likelihood : $y_i \vert \theta \sim B(n,\theta)$, $\;\;i=1,2,3,\cdots,n$

Prior : $\theta \sim Beta(\alpha, \beta)$

Posterior를 구해보면 다음과 같다.

$$
\begin{align*}
p(\theta \vert y) &\propto p(y \vert \theta)p(\theta)\\
&\propto \prod_{i=1}^n p(y_i \vert \theta) p(\theta)\\
&\propto \binom{n}{y_i} \theta^{y}(1-\theta)^{n-y} {\Gamma{(\alpha+\beta)} \over {\Gamma{(\alpha)}\Gamma{(\beta)}}}\theta^{\alpha-1}(1-\theta)^{\beta-1}\\
&\propto \theta^{y + \alpha - 1}
(1-\theta)^{n-y+\beta-1}\\
&\propto \theta^{(y + \alpha) - 1}
(1-\theta)^{(n-y+\beta)-1}\\
&\sim Beta(y + \alpha,n-y+\beta)
\end{align*}
$$

<br>
<br>
### Example 2- Conjugate prior

Likelihood : $y_i \vert \theta \sim Pois(\theta)$, $\;\;i=1,2,3,\cdots,n$

Prior : $\theta \sim Gamma(\alpha, \beta)$

Posterior를 구해보면 다음과 같다.

$$
\begin{align*}
p(\theta \vert y) &\propto p(y \vert \theta)p(\theta)\\
&\propto \prod_{i=1}^n p(y_i \vert \theta) p(\theta)\\
&\propto \prod_{i=1}^n {e^{-\theta}\theta^{y_i} \over {y_i!}} { \beta^{\alpha} \over \gamma{(\alpha)}}\theta^{\alpha-1}e^{- \beta \theta}\\
&\propto \theta^{\sum_{i=1}^n y_i + \alpha - 1}
e^{-n\theta-\beta \theta}\\
&\propto \theta^{(\sum_{i=1}^n y_i + \alpha) - 1}
e^{-(n+\beta) \theta}\\
&\sim Gamma(\sum_{i=1}^n y_i + \alpha,n+\beta )
\end{align*}
$$

<br>
<br>
### Example 3- Conjugate prior

Likelihood : $y_i \vert \theta \sim N(\mu, \theta)$, $\;\;i=1,2,3,\cdots,n$

Prior : $\theta^{-1} \sim Gamma(\alpha, \beta)$

Posterior를 구해보면 다음과 같다.

$$
\begin{align*}
p(\theta \vert y) &\propto p(y \vert \theta)p(\theta)\\
&\propto \prod_{i=1}^n p(y_i \vert \theta) p(\theta)\\
&\propto \prod_{i=1}^n {
  1 \over {\sqrt{2\pi\theta}}} e^{- {1 \over 2\theta}(y_i-\mu)^2} { \beta^{\alpha} \over \gamma{(\alpha)}}\theta^{-\alpha-1}e^{- \beta /\theta}\\
&\propto \theta^{- {1 \over 2} n}  e^{- {1 \over 2\theta}\sum_{i=1}^n(y_i-\mu)^2}\theta^{-\alpha-1}e^{- \beta /\theta}\\
&\propto \theta^{- {1 \over 2} n-\alpha-1}e^{- {1 \over 2\theta}\sum_{i=1}^n(y_i-\mu)^2- \beta /\theta}\\
&\propto \theta^{- ({1 \over 2} n+\alpha)-1}e^{- ({1 \over 2}\sum_{i=1}^n(y_i-\mu)^2+ \beta) /\theta}\\
&\sim Inv-Gamma({1 \over 2} n+\alpha, {1 \over 2}\sum_{i=1}^n(y_i-\mu)^2+ \beta)
\end{align*}
$$



<br>
<br>
