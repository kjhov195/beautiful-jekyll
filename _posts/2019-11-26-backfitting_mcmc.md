---
layout: post
title: Backfitting MCMC
subtitle: MCMC
category: Bayesian
use_math: true
---

<br>
<br>
### Backfitting MCMC

Given the observed data $y$, our bayesian setup induces a posterior distribution

$$ p((T_1, M_1), \dots, (T_m,M_m) \vert y)$$

At a general level, our algorithm is a Gibbs sampler.

Let
$$
\begin{align*}
\begin{cases}
T_{(j)} &:\text{ the set of all trees in the sum except  } T_j\\
&\;\;\;\rightarrow \text{a set of (m-1) trees}\\
M_{(j)} &: \text{the associated terminal node parameter}
\end{cases}
\end{align*}
$$

<br>

$$
\begin{align*}
\Rightarrow \text{Conditional distribution:}\\\\
(T_j,M_j)\vert T_{(j)},M_{(j)},\sigma,y\;\;\;\;\cdots(1)\\
\sigma \vert T_1,\cdots,T_m,M_1,\cdots,M_m,y\;\;\;\;\cdots(2)\\\\
where\;j=1,2,\cdots,m
\end{align*}
$$

<br>

(2) draws from an inverse-gamma distribution

<br>

(1) Let $\beta_j \equiv y- \sum_{k \neq j} g(x;T_k,M_k) = g(x;T_j,M_j)+\epsilon$

Drawing from $\;(T_j,M_j)\vert T_{(j)},M_{(j)},\sigma,y\;$ is equivalent to

drawing from $\;(T_j,M_j)\vert \beta_j,\sigma,\;\;j=1,2,\cdots,m$

<br>

$$
\begin{align*}
P((T_j,M_j)\vert \beta_j,\sigma) = P(T_j \vert \beta_j, \sigma) P(M_j \vert T_j, \beta_j, \sigma)
\end{align*}
$$

<br>

$P(T_j \vert \beta_j, \sigma) \propto P(T_j) \int P(\beta_j \vert M_j, T_j, \sigma) P(M_j \vert T_j, \sigma) dM_j$

can be obtained using Metropolis-Hastings(MH) algorithm of CGM98(Hugh A. Chipman, Edward I. George and Robert E. McCulloch(1998), Bayesian CART Model Search) and priors explained previously.

<br>

$P(M_j \vert T_j, \beta_j, \sigma)$

a set of independent draws of the terminal node $\mu_{ij}$'s from a normal distribution.



<br>
<br>
### Reference

Antonio R. Linero(2016), Bayesian Regression Trees for High-Dimensional Prediction and Variable Selection, Journal of the American Statistical Association

Hugh A. Chipman, Edward I. George and Robert E. McCulloch(1998), Bayesian CART Model Search, Journal of the American Statistical Association

<br>
<br>
