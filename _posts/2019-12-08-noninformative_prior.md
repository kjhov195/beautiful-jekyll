---
layout: post
title: Non-informative Prior(Flat/Jeffreys Prior)
subtitle: Bayesian Statistics
category: Bayesian
use_math: true
---

<br>
<br>
### Non-informative Prior

앞서 Conjugate Prior에 대해 살펴보았다. Prior는 $\theta$의 분포에 대한 믿음으로 볼 수 있는데, Prior를 잘못 설정하게 되면 잘못된 결과로 이어지게 될 가능성이 크다. Prior를 잘못 정했을 때의 문제점을 해결하는 방법으로는 Prior가 Posterior에 영향을 미치는 information을 약하게 만들고, Likelihood에 강하게 주는 방법이 있을 것이다.

$$ p(\theta \vert y) \propto p(y \vert \theta) p(\theta)$$

<br>
<br>
### Flat Prior

Prior distribution을 flat하게 가정하는 방법을 Flat prior라고 하며, vague/diffuse/reference prior라고도 한다.

$$ p(\theta ) \propto 1 $$

$p(\theta) \propto 1$을 가정할 경우 posterior는 다음과 같이 계산된다.

$$
\begin{align*}
p(\theta \vert y) &\propto p(y \vert \theta) p(\theta)\\
&\propto p(y \vert \theta)
\end{align*}
$$

<br>

Flat prior를 가정할 경우 prior가 Posterior에 주는 영향을 상당히 작게 만들 수 있는 장점이 있다. 즉, Prior의 영향을 매우 작게 만들어 Likelihood의 영향을 상대적으로 크게 만들어줌으로써 Data 그 자체에 의존하게 만들어 줄 수 있다.

$$Let\;the\;data\;speak\;for\;themselves$$

<br>

하지만 Practical하게는 사용되지 않는데, 여러가지 이유가 있겠지만 가장 큰 이유는 $p(\theta) \propto 1$을 가정할 경우 $\int_{-\infty}^\infty p(\theta) d\theta = \infty$가 되므로 분포라고 보기 어렵기 때문이다.

또 하나의 이유는 transformation에 대하여 invariance가 부족하다는 점이다. 다음의 예를 살펴보자.

$\theta$의 prior를 flat prior로 가정해보자. 그리고 $\theta$에 exponential을 취해주는 transformation해준 모수를 $\psi$라고 하자.($\psi = exp(\theta)$)

$\theta$의 prior는 flat prior를 가정하여 $p(\theta) \propto 1$이라고 하자. 하지만 $\psi$의 prior는 다음에서 볼 수 있듯이 더 이상 flat prior가 아니다. 즉, flat prior를 가정할 경우 transformation에 invariance한 성질이 없는 것을 확인할 수 있다.

$$
\begin{align*}
p(\psi) &= p(\theta) \left | \frac{d\theta}{d\psi} \right \vert \quad\\
&\propto p(\theta) {1 \over \psi}\\
&\propto {1 \over \psi} \neq 1
\end{align*}
$$


<br>
<br>
### Jeffreys non-informative Prior

one-to-one Transformation에 대하여 invariance한 성질을 가질 수 있도록 만들어주는 non-informative prior가 바로 Jeffreys non-informative prior이며, 다음과 같다.

$$ p(\theta) \propto \sqrt{det I(\theta)}$$

prior를 이렇게 설정하면 one-to-one transformation에 대해 invariant한 성질을 가지게 되는데 이를 증명해보면 다음과 같다.

$$
\begin{align*}
p(\theta) &\propto \sqrt{det I(\theta)}\\
\psi &= f(\theta)\;\;\;\;\text{for some transformation f}\\\\
p(\psi) &= p(\theta)\left | det\frac{d\theta}{d\psi} \right \vert \quad\\
&\propto
\sqrt{det {\partial \theta_k \over \partial \psi_i}}
\sqrt{det E \left \lbrack {-{\partial^2 logL(\theta \vert y) \over \partial \theta_k \partial \theta_l}} \right \rbrack}
\sqrt{det {\partial \theta_l \over \partial \psi_k}}\\
&= \sqrt{det E \left \lbrack {-{\partial^2 logL(\theta \vert y) \over \partial \psi_i \partial \psi_j}} \right \rbrack}\\
&= \sqrt{det I(\psi)}
\end{align*}
$$


<br>
<br>
