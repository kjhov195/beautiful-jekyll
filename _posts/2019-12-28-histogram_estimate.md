---
layout: post
title: Histogram estimate
subtitle: Nonparametric Density Estimation
category: Statistics
use_math: true
---

<br>
<br>
### Histogram estimate

Histogram estimate는 말 그대로 histogram을 통하여 pdf $f$를 estimate하는 방법이다. Histogram estimate는 다음 3가지 특성을 가진다.

1. __locations__ 와 __widths of the bins__ 에 큰 영향을 받는다.

2. __locations__ 와 __widths of the bins__ 가 적절하게 선택이 잘 된다면, performance가 꽤 좋다.

3. Blocky하고, discontinuous하다.

<br>
<br>
### Example: Yellowstone National Park dataset

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

### Reference

Sangun Park(2019), Nonparametric Statistics: Nonparametric Density Estimation, Yonsei University
