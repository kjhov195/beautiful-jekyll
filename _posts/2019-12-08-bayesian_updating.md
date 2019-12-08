---
layout: post
title: Bayesian Updating
subtitle: Bayesian Statistics
category: Bayesian
use_math: true
---

<br>
<br>
앞서 Posterior distribution에 대하여 살펴보았다. 우리는 Posterior는 Likelihood와 prior의 곱에 비례한다는 것을 알고 있다.

$$ p(\theta \vert y) \propto p(y \vert \theta) p(\theta)$$

<br>
<br>
### Bayesian updating

우리가 가장 최근에 얻은 data를 Current data라고 하자.

__Current data를 얻기 이전__ 에 가지고 있는 $\theta$에 대한 정보를 우리는 prior라고 한다. 이러한 prior와 __Current data에 들어있는__ 정보인 Likelihood를 Combine하여 $\theta$에 대한 정보를 Update할 수 있다.

데이터 y가 주어진 뒤, Prior와 Likelihood를 결합함으로써 얻은 새로운 $\theta$에 대한 믿음(Belief)이 생기게 되면, 이는 다시 새로운 Prior가 된다.

즉 새로운 정보를 얻었을 때 Posterior distriubtion이 다음 계산에서의 Prior로 사용되는데, 이를 Bayesian Updating이라고 한다. 수식으로 살펴보면 이해가 조금 더 쉽다.

$$
\begin{align*}
\text{1. 1st Step:}\;\;\;\;\;\\
p(\theta \vert y_1) &\propto p(y_1 \vert \theta) p(\theta)\\\\
\text{2. 2nd Step:}\;\;\;\;\;\\
p(\theta \vert y_1, y_2) &\propto p(y_1,y_2 \vert \theta) p(\theta)\\
&= p(y_1 \vert \theta) p(y_2 \vert \theta) p(\theta)\;\;\;\;\;\;(y_1\;\text{&}\;y_2\;are\;indep)\\
&\propto p(y_2 \vert \theta) p(\theta \vert y_1)
\end{align*}
$$

<br>
<br>
### Example

어떠한 감기의 발병 확률이 20%라고 한다. 감기 진단의 정확률은 다음과 같다.

실제로 감기 환자일 때, 의사가 정확하게 감기라고 진단할 확률은 95%이고,

실제로 감기 환자가 아닐 때, 의사가 정확하게 감기가 아니라고 진단할 확률은 90%라고 한다.

<br>

$$
\begin{align*}
p(Cold) &= 0.2\\
P(Positive \vert Cold) &= 0.95\\
P(Negative \vert not\;Cold) &= 0.90
\end{align*}
$$

<br>

만약 첫 번째 진료에서 Positive(양성) 반응이 나왔을 때 실제 감기일 확률을 계산해 보면 다음과 같다.(계산은 간단하게 Bayes Rule만 사용하면 된다.)

<br>

$$
\begin{align*}
p(Cold \vert Positive) &= {p(Positive \vert Cold) p(Cold) \over p(Positive)}\\
&= {p(Positive \vert Cold) p(Cold) \over p(Positive, Cold)+p(Positive, not\;Cold)}\\
&= {p(Positive \vert Cold) p(Cold) \over p(Positive \vert Cold) p(Cold)+p(Positive \vert not\; Cold)p(not\;Cold)}\\
&= {p(Positive \vert Cold) p(Cold)}\over {p(Positive \vert Cold)p(Cold)+(1-p(Negative \vert not\; Cold))p(not\;Cold)}\\
&= {0.95 \cdot 0.2 \over 0.95 \cdot 0.2 + (1-0.9)\cdot0.8 }\\
&\approx 0.7037
\end{align*}
$$



<br>
<br>
