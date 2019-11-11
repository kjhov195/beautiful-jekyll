---
layout: post
title: Empirical Bayes Estimate
subtitle: Bayesian Statistics
category: Statistics
use_math: true
---

### Related Post
[Hierarchical linear model (1)](https://kjhov195.github.io/2019-11-02-hierarchical_linear_model_1/)

[Hierarchical linear model (2)](https://kjhov195.github.io/2019-11-03-hierarchical_linear_model_2/)

[Empirical Bayes Estimate](https://kjhov195.github.io/2019-11-11-Empirical_Bayes/)

---

<br>
<br>


### Bayes Rule

Empirical Bayes Estimates에 대해 살펴보기 전에, 간단하게 Bayesian Statistics에 대해 살펴보겠다. Bayesian은 모든 이야기가 Bayes rule에 기반하여 이루어진다. Bayes rule을 다시 떠올려보면, 다음과 같다.

$$
\begin{align*}
p(A|B) = {p(B|A)p(A) \over p(B)}
\end{align*}
$$

<br>
<br>
### Bayesian Statistics

Bayesian에서는 모수 $\theta$를 unkown fixed value로 생각하는 것이 아닌, 어떠한 분포를 가지는 확률변수로 생각한다. 모수 $\theta$의 분포를 ___prior distribution___ 이라고 한다.

반면, y가 given일 때, $\theta$의 분포를 ___posterior distribution___ 이라고 한다.

확률변수 Y의 분포가 모수 $\theta$를 가지는 어떠한 분포를 따른다고 하자. 만약 $\theta$ 또한 또 다른 하나의 확률변수로 생각한다면, 이 떄 Y와 $\theta$의 Joint distribution은 Bayes rule을 활용하여 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
f(y, \theta) &= f(y | \theta)f(\theta)\\
&= {f(\theta | y)f(y) \over f(\theta)} f(\theta)\\
&= f(\theta | y)f(y)
\end{align*}
$$

따라서 다음과 같이 ___posterior distribution___ 을 얻을 수 있다.

$$
\begin{align}
f(\theta|y) &= {f(y|\theta)f(\theta) \over f(y)}\\
&\propto f(y|\theta)f(\theta)
\end{align}
$$

<br>
<br>

### Bayes Estimator

이 posterior distribution의 평균(___posterior mean___)이 $\theta$ 의 추정량으로 종종 사용되는데, 이를 Bayes Estimator라고 한다.

<br>
<br>
### Empirical Bayes Estimator

식 (1)을 y의 pdf에 대해 정리하면 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
f(y) = {f(y|\theta)f(\theta) \over f(\theta|y)}
\end{align*}
$$

앞서 말했듯이 Bayesian에서는 $\theta$를 constant가 아닌, random variable로 생각한다. 를 prior distribution,  를 posterior distribution이라고 한다.

$\theta$의 __Prior distribution: $p(\theta)$__ 와 __Posterior distribution: $p(\theta|y)$__ 이 또 하나의 모수 $\phi$를 가진다고 해보자. 그렇다면, y의 pdf를 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
f(y) = {f(y|\theta)f(\theta|\phi) \over f(\theta|y,\phi)}
\end{align*}
$$

이 때 주어진 $y$를 활용하여 $\phi$를 추정할 수 있는 경우가 존재한다. 이렇게 구한 $\hat \phi$를 사용하여 새로운 __Posterior distribution: $p(\theta|y,\hat \phi)$__ 을 얻을 수 있다.

$\phi$를 추정한 뒤, 새롭게 구한 posterior distribution의 평균(__posterior mean__)을 $\theta$의 추정량으로 사용할 수 있는데, 이를 Empirical Bayes Estimator라고 한다.

즉, Empirical Bayes Estimator는 prior distribution을 임의로 가정하는 기존의 Bayesian과는 달리, 데이터로부터 prior distribution의 parameter를 추정하여 prior distribution을 설정하게 된다. 이러한 방식을 활용하여 posterior mean을 구하고, 이를 $\theta$의 추정량으로 사용하는 것을 Empirical Bayes Estimator라고 한다.

Empirical이라는 단어가 붙은 것은 임의로 prior distribution을 설정해주는 것이 아닌 경험적으로(Empirically), 즉 data를 활용하여 prior의 모수를 추정하고, prior distribution을 설정해주기 때문에 붙은 이름이라고 볼 수 있다.

<br>
<br>
### 2. Example

간단한 example을 통하여 Empirical Bayes Estimates를 살펴보도록 하겠다.

$Y_i$가 $pois(\lambda_i)$를 따른다고 하자. poisson distribution의 경우 모수의 support가 양수이므로, prior distribution으로 Gamma를 줄 수 있다. 즉, $\lambda_i$가 $Gamma(\alpha, \beta)$를 따른다고 하자.($\alpha$는 known이다.)

$$
\begin{align*}
\\
Let\; Y_i &\overset{\text{iid}}{\sim} Pois(\lambda_i), \; i=1,...,n\\
\lambda_i &\overset{\text{iid}}{\sim} Gamma(\alpha, \beta)\;\;\;\; where\;\alpha:\;known,\;\beta:\;unknown\\\\
\end{align*}
$$

우리는 $\beta$를 임의로 정해주는 것이 아닌, data를 활용하여 estimate하려고 한다.

$$
\begin{align*}
\\
f(Y_i|\beta)
&= \int_0^{\infty} \left[ \frac {\beta^\alpha}{\Gamma(\alpha)} \lambda_i^{\alpha-1} e^{-\beta \lambda_i} \right] \left[ \frac {e^{-\lambda_i}\lambda_i^{y_i}}{y_i {!}} \right] d\lambda_i\\
&= \frac {\beta^\alpha}{{y_i {!}}\Gamma(\alpha)}  \int_0^{\infty} \left[ \lambda_i^{\alpha-1} e^{-\beta \lambda_i} \right] \left[ {e^{-\lambda_i}\lambda_i^{y_i}} \right] d\lambda_i\\
&= \frac {\beta^\alpha}{{y_i {!}}\Gamma(\alpha)}  \int_0^{\infty} \lambda_i^{y_i+\alpha-1} {e^{-(\beta+1)\lambda_i}}  d\lambda_i\\
&= \frac {\beta^\alpha}{{y_i {!}}\Gamma(\alpha)} \frac {\Gamma(y_i+\alpha)}{(\beta+1)^{y_i+\alpha}}\;\;(\because gamma\;kernel\;\Gamma(y_i+\alpha,\beta+1))  \\
&= \binom{y_i+\alpha-1}{\alpha-1} \left(\frac {\beta}{\beta+1} \right)^\alpha \left(\frac {1}{\beta+1} \right)^{y_i} \sim Negative\; Binomial
\end{align*}
$$

Likelihood를 구해보자.

$$
\begin{align*}
L(Y_i|\beta) &= \prod_{i=1}^n \left[\binom{y_i+\alpha-1}{\alpha-1} \left(\frac {\beta}{\beta+1} \right)^\alpha \left(\frac {1}{\beta+1} \right)^{y_i}\right] \\
&= \left[ \prod_{i=1}^n\binom{y_i+\alpha-1}{\alpha-1}\right]\left(\frac {\beta}{\beta+1} \right)^{n\alpha} \left(\frac {1}{\beta+1} \right)^{\sum y_i}\\\\
l &= \log(L(Y_i|\beta))
\end{align*}
$$

$l$을 maximize하는 $\hat \beta_{mle}$를 구해보면 $\hat \beta_{mle} = \frac {\alpha}{\overline {Y}}$임을 알 수 있다.

이제 Empirical Bayes Estimator를 구해보자. prior, posterior distribution for $\lambda_i$는 다음과 같다.

$$
\begin{align*}
\\
\lambda_i &\sim Gamma(\alpha,\hat  \beta)\\
&= Gamma(\alpha,\frac {\alpha}{\overline {y}})\;\;:prior\\\\
\lambda_i|{y_i,\hat \beta} &\sim Gamma(y_i+\alpha, 1+\hat \beta)\;\;:posterior\\\\
\end{align*}
$$

posterior mean을 다음과 같이 구할 수 있고, 이를 $\lambda_i$에 대한 추정량으로 사용할 수 있다.(Empirical Bayes Estimator)

$$
\begin{align*}
(posterior\;mean) &= (Y_i+\alpha)/(1+\hat \beta)\\
&= (Y_i+\alpha)/(1+\alpha/{\overline Y})\\
&= \left(\frac {\overline Y}{\overline Y + \alpha}\right) \left(Y_i+\alpha \right)
\end{align*}
$$

<br>
<br>
### 3. Hierarchical model에서...

writing...
