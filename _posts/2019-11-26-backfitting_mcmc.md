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
\begin{cases}
T_{(j)} &:\text{ the set of all trees in the sum except  } T_j\\
&\;\;\;\rightarrow \text{a set of (m-1) trees}\\
M_{(j)} &: \text{the associated terminal node parameter}
\end{cases}
$$

<br>

$$
\begin{align}
\Rightarrow \text{Conditional distribution:}\\\\
(T_j,M_j)\vert T_{(j)},M_{(j)},\sigma,y\\
\sigma \vert T_1,\cdots,T_m,M_1,\cdots,M_m,y
\end{align}
$$

<br>
<br>

### Reference

Antonio R. Linero(2016), Bayesian Regression Trees for High-Dimensional Prediction and Variable Selection, Journal of the American Statistical Association

<br>
<br>
