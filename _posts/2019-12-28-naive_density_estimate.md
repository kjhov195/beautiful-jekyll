---
layout: post
title: Naive Density Estimate
subtitle: Nonparametric Density Estimation
category: Statistics
use_math: true
---

<br>
<br>
### 1. Histogram Estimate

Histogram estimate는 말 그대로 histogram을 통하여 pdf $f$를 estimate하는 방법이다. Histogram estimate는 다음 3가지 특성을 가진다.

1. __locations__ 와 __widths of the bins__ 에 큰 영향을 받는다.

2. __locations__ 와 __widths of the bins__ 가 적절하게 선택이 잘 된다면, performance가 꽤 좋다.

3. Blocky하고, discontinuous하다.

<br>
<br>
### Example: Yellowstone National Park dataset

<br>

<center><img src = '/post_img/191228/image3.png' width="300"/></center>

bins = 8일 때의 histogram이다.

<br>

<center><img src = '/post_img/191228/image4.png' width="300"/></center>

bins = 20일 때의 histogram이다.

<br>

<center><img src = '/post_img/191228/image5.png' width="300"/></center>

bins = 30일 때의 histogram이다.

<br>
<br>
### 2. Naive Density Estimate

pdf $f$를 추정하는 또 다른 방법으로 Empirical Density Estimate를 확장시킨 Naive Density Estimate가 있다. 우리는 다음과 같이 cdf $F$에 대한 derivative로 pdf $f$를 구할 수 있다는 사실을 알고 있다.

$$ f(x) = \lim_{h \to 0} {{F(x+h)-F(x-h)} \over 2h}$$

이 사실을 활용하여 h를 아주 작은 어떠한 값으로 정한다면 우리는 Empirical Density Estimate $\hat {F(x)}$를 활용하여 다음과 같이 $f(x)$를 $\hat {f(x)}$로 추정할 수 있다.

$$
\begin{align*}
\hat f(x) &\equiv {1 \over {2h}} \left( \hat F_n(x+h) - \hat F_n(x-h) \right)\\
& = {1 \over {2nh}} \left( \sum_{j=1}^n I(x-h < x_j \leq x+h) \right)\\
& = {1 \over n} \left( \sum_{j=1}^n  {1 \over h} {1 \over 2} I(-1 < {{x-x_j} \over h} \leq 1) \right)\\
& = {1 \over n} \left( \sum_{j=1}^n  {1 \over h} K({{x-x_j} \over h}) \right)\\
where\;\;K(x)&= { 1 \over 2 } \cdot I(-1 < x \leq 1)
\end{align*}
$$

<br>
<br>
### Naive Density Estimator Souce code

```
def indicator(x):
    if -1<x<=1:
        result = 1
    else:
        result = 0

    return result

def K(x):
    result = 1/2*indicator(x)

    return result

def naive_density_estimate(x, h, data):
    n = len(data)
    summation = []
    for i in range(n):
        summation.append(K((x-data[i])/h)/h)
    result = 1/n*sum(summation)

    return result
```

<br>
<br>
### Example: Yellowstone National Park dataset

<br>

<center>
<img src = '/post_img/191228/image6.png' width="250"/>
<img src = '/post_img/191228/image7.png' width="250"/>
<img src = '/post_img/191228/image8.png' width="250"/>
</center>

<center>
<img src = '/post_img/191228/image9.png' width="250"/>
<img src = '/post_img/191228/image10.png' width="250"/>
<img src = '/post_img/191228/image11.png' width="250"/>
</center>

각각 왼쪽 위부터 h = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6일 때 Naive Density Estimate의 결과이다.

우리가 이 예시를 통해 확인할 수 있는 Naive Density Estimate의 단점이 존재하는데, 위 결과에서 볼 수 있듯이 추정된 pdf가 continuous하지 않으며, 상당히 ragged되어 있는 모습을 확인할 수 있다.(The naive density estimator is not continuous and will have a ragged pattern.)


<br>
<br>
### Reference

Sangun Park(2019), Nonparametric Statistics: Nonparametric Density Estimation, Yonsei University
