---
layout: post
title: Hierarchical linear model (2)
subtitle: Empirical Bayes Estimate
category: Statistics
use_math: true
---

### 1. Estimating level 1 coefficient

앞서 Hierarchical model의 기본적인 형태에 대하여 간략하게 살펴보았다. 이제 Hierarchical model에서 1 level coefficients를 estimate하는 방법을 살펴보도록 하자.

앞서 살펴본 학생들의 SES-수학성적의 예제에서는 다음과 같은 two-level model을 세워보았다.

$$
\begin{align*}
Y_{ij} &= \beta_{0j} + \beta_{1j} (X_{ij}-\overline{X}_{\cdot j}) + \epsilon_{ij}\\
&= (\gamma_{00}+\gamma_{01}W_j+u_{0j}) + (\gamma_{10}+\gamma_{11}W_j+u_{1j})(X_{ij}-\overline{X}_{\cdot j})+\epsilon_{ij}\\
&= \gamma_{00}+\gamma_{01}W_j+\gamma_{10}(X_{ij}-\overline{X}_{\cdot j})+\gamma_{11}W_j(X_{ij}-\overline{X}_{\cdot j})+u_{0j}+u_{1j}(X_{ij}-\overline{X}_{\cdot j})+\epsilon_{ij\cdot}\\\\
where\;
\begin{bmatrix}
\beta_{0j} \\
\beta_{1j}
\end{bmatrix} &\sim N
\begin{pmatrix}
\begin{bmatrix}
\gamma_0 \\
\gamma_1
\end{bmatrix}
,
\begin{bmatrix}
\tau_{00} & \tau_{01} \\
\tau_{01} & \tau_{11}
\end{bmatrix}
\end{pmatrix}\\\\
\beta_{0j} &= \gamma_{00}+\gamma_{01}W_j+u_{0j}\\
\beta_{1j} &= \gamma_{10}+\gamma_{11}W_j+u_{1j}\\\\
W_j &=
\begin{cases}
1 \;\;\; if\;School\;j\;is\;Mission\;school\\
\\
0 \;\;\;\; if\;School\;j\;is\;Public\;school\\
\end{cases}\\\\
\begin{bmatrix}
u_{0j} \\
u_{1j}
\end{bmatrix} &\sim
\begin{pmatrix}
\begin{bmatrix}
0 \\
0
\end{bmatrix}
,
\begin{bmatrix}
\tau_{00} & \tau_{01} \\
\tau_{01} & \tau_{11}
\end{bmatrix}
\end{pmatrix}\\\\
\end{align*}\\
$$

level-1-model을 sample mean의 형태로 나타내면 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
\overline {Y_{\cdot j}} &= \beta_{0j} + \overline{\epsilon_{\cdot j}}\\\\
where\;\;\overline{\epsilon_{\cdot j}} &\sim N(0,V_j)\\
V_j &= \sigma^2/n_j
\end{align*}
$$

마찬가지로, level-2-model을 sample mean의 형태로 나타내면 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
\beta_{0j} = \gamma_{00}+u_{0j}\\\\where\;u_{0j} \sim N(0, \tau_{00})
\end{align*}
$$

이 모델에서는 두 가지 alternative estimators for $\beta_{0j}$가 존재한다.

#### a. level 1 coefficient를 추정하는 방법

##### 첫 번째 방법 : simple approach

$$\overline {Y_{\cdot j}} = \beta_{0j}+\overline{\epsilon_{\cdot j}}$$

level-1 model 모델에서 우리는 $\beta_{0j}$에 대한 unbiased estimator가 $\overline {Y_{\cdot j}}$임을 알 수 있다.

$$ \hat{\beta_{0j}} = \overline {Y_{\cdot j}}$$

$$ E[\overline {Y_{\cdot j}}] = \beta_{0j}$$

$$ Var[\overline {Y_{\cdot j}}] = \tau_{00} + V_j$$

##### 두 번째 방법 : bayes estimator(Lindley & Smith, 1972)


$$
\begin{align*}
\hat{\beta_{0j}}^* = {\sum {\Delta_j}^{-1} \overline {Y_{\cdot j}}} \over {\sum {\Delta_j}^{-1}}
\end{align*}
$$

우리는 Bayes Estimator $\hat{\beta_{0j}}^* $를 또 다른 Estimator로 생각할 수 있다.

Bayes Estimator는 다음과 같은 두 부분의 적절한 weigted sum(weighted combination)으로 이루어진 optimal한 point에서 구할 수 있는 Estimator이다.

$$ \hat{\beta_{0j}}^* = \sum \lambda_j \overline {Y_{\cdot j}} + (1-\lambda_j) \hat {\gamma_{00 \cdot}} $$

여기서 $\lambda_j$를 $\hat{\beta_{0j}}_{LSE} =\overline {Y_{\cdot j}}$ 의 ___reliablility___ 라고 한다.(Kelley, 1927)

$\lambda_j$(Reliability)는 다음과 같이 얻을 수 있다.


$$
\begin{align*}
\lambda_j &= {Var(\beta_{0j}) \over Var(\overline {Y_{\cdot j}})}\\
&= {\tau_{00} \over (\tau_{00} + V_j)}
\end{align*}
$$

<br>
#### b. level 1 coefficient : SES-수학성적 Example

writing...

<br>
#### c. $\lambda_j$ (reliability)

우리는 $\lambda_j$를 ___reliablility___ 라고 부른다. 그 이유가 뭘까?

위에서 살펴보았듯이, $\lambda_j$의 정의는 다음과 같다.

$$
\begin{align*}
\lambda_j &= {Var(\beta_{0j}) \over Var(\overline {Y_{\cdot j}})}\\\\
&= {\tau_{00} \over (\tau_{00} + V_j)}\\\\
&= {(parameter\;variance) \over (total\;variance)}\\\\
&= {(true\;score) \over (observed\;score)}
\end{align*}
$$

전통적인 통계학의 관점에서 $\overline {Y_{\cdot j}}$를 구하는 것은 unknown parameter $\beta_{0j}$ 를 'measure'하는 것이라고 볼 수 있다.

마찬가지로 $\lambda_j$를 구하는 것은 $\overline {Y_{\cdot j}}$의 parameter variance와 total variance의 비율를 'measure'하는 것이라고 볼 수 있다.

만약 $\lambda_j \approx 1$이라면 어떤 의미로 해석할 수 있을까?

$\lambda_j \approx 1$인 경우는 다음 두 가지 경우 중 하나로 생각할 수 있다.

- constant sample size per group을 가정하였을 때, group means($\beta_{0j}$)이 level-2 units간 상당한 차이를 보이는 경우  

- sample size $n_j$ 가 충분히 큰 경우

만약 sample mean이 'highly reliable estimate'라면($\lambda_j \approx 1$ 이라면) $\overline {Y_{\cdot j}}$에 큰 weight를 주어 $\beta_{0j}$를 추정하게 된다. ,

반면, sample mean이 'unreliable'하다면($\lambda_j \approx 0$ 이라면) $\hat {\tau_{00}}$에 큰 weight를 주어 $\beta_{0j}$를 추정하게 된다.

한편, 식의 변형을 통하여 $\lambda_j$에 대해 다음과 같이 생각해 볼 수도 있다.

$$
\begin{align*}
\\\\
\lambda_j &= {Var(\beta_{0j}) \over Var(\overline {Y_{\cdot j}})}\\\\
&= {\tau_{00} \over (\tau_{00} + V_j)}\\\\
&= {V_j^{-1} \over (V_j^{-1} + \tau_{00}^{-1})}\\\\
1-\lambda_j &= 1 - {V_j^{-1} \over (V_j^{-1} + \tau_{00}^{-1})}\\
&= {\tau_{00}^{-1} \over (V_j^{-1} + \tau_{00}^{-1})}
\end{align*}
$$

즉, $\hat{\beta_{0j}}^* $를 구성하는 $\overline {Y_{\cdot j}}$에 대한 weight는 $V_j^{-1}$에 proportional하다고 볼 수 있다.

<br>
#### d. $ \beta_{0j}^* $ is optimal 하다는 것의 의미

writing...

<br>
<br>
### 2. Empirical Bayes Estimate
#### a. Definition

writing...

<br>
#### b. 왜 이런 경우에 Empirical Bayes Estimate가 되는가?

writing...

<br>
<br>
### Reference
Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical linear models: Applications and data analysis methods (Vol. 1). Sage. 45-47
<br>
<br>
<br>
