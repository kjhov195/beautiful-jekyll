---
layout: post
title: Prior/Posterior
subtitle: Bayesian Statistics
category: Bayesian
use_math: true
---

<br>
<br>
### Frequentist vs Bayesian

Frequentist의 경우, 모수(parameter)를 알 수 없는 고정된 수, __Constant__ 로 취급한다. 따라서 이러한 constant의 모수를 전제로 하여, 데이터(y)가 주어졌을 때의 Likelihood, $L(\theta \vert y)$를 계산하여 이를 Maximize하는 $\theta$를 찾는다. 이렇게 찾은 $\theta$의 추정치가 mle of $\theta$가 된다.

$$\hat \theta_{mle} = Argmax_{\theta} L(\theta \vert y)$$

반면 Bayesian의 경우, 모수를 __Random Variable__ 로 취급한다. 이것이 가장 핵심적인 차이점이다.

베이지안은 모수에 대한 분포를 가정하며, 이를 __Prior__ distribution이라고 한다. 직관적으로 이해해 본다면 posterior는 관찰하기 이전의 믿음 정도로 이해할 수 있을 것이다.

또한 관측된 데이터에 대한 모수의 Conditional Distribution이라고 할 수 있는 __Posterior__ Distribution를 통해 추론이 이루어진다.

<br>
<br>
### Bayes Rule

Bayesian은 모든 이야기가 Bayes rule에 기반하여 이루어진다. Bayes rule을 다시 떠올려보면, 다음과 같다.

$$
\begin{align*}
p(A|B) = {p(B|A)p(A) \over p(B)}
\end{align*}
$$

<br>
<br>
### Prior

Bayesian에서는 모수 $\theta$를 unkown constant로 생각하는 것이 아닌, 어떠한 분포를 가지는 확률변수로 생각한다. 모수 $\theta$의 분포를 ___prior distribution___ 이라고 한다.

$$p(\theta) \sim prior\;dist\;of\;\theta$$

<br>
<br>
### Posterior

반면, y가 given일 때, $\theta$의 분포를 ___posterior distribution___ 이라고 한다.

$$
\begin{align*}
p(\theta \vert y) &= {p(y \vert \theta)f(\theta) \over p(y)}\\
&\propto p(y \vert \theta)p(\theta)\\\\
Posterior &\propto Likelihood \cdot Posterior\\\\
\end{align*}
$$

<br>
<br>
