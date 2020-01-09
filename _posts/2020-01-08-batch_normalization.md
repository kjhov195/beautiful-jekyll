---
layout: post
title: Batch Normalization(BN)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>
### Vanishing Gradient

우리는 앞서 ReLU 등의 [변형된 Activation 함수를 사용](https://kjhov195.github.io/2020-01-07-activation_function_2/)하거나, [Weight initialization를 신중하게 하는 방법](https://kjhov195.github.io/2020-01-07-weight_initialization/)을 통하여 Vanishing gradient 문제를 간접적으로 해결해 보았다.

이번에는 학습하는 과정 자체를 개선하여 근본적으로 Vanishing Gradient 문제가 발생하지 않도록 하는 방법에 대하여 살펴보도록 하겠다.

<br>
<br>
### Batch Normalization


Batch Normalization은 loffe and Szegedy(2015)에 의하여 제시된 아이디어이다.

일반적인 Neural Networks에서는 여러 layers를 사용한다. 그러한 각 layer마다 input을 받아 linear combination을 구한 후, Activation function을 적용하여 output을 구해주는 작업이 이루어 진다.

<br>

<center><img src = '/post_img/200108/image7.png' width="600"/></center>

결과적으로 이 때문에 각 layer의 input data $x$의 분포(Distribution)가 달라지게 된다. 각 layer의 input data 분포의 변형은 layer를 지나감에 따라 누적되어 최종적인 output의 분포는 상당히 많이 달라지게 된다.

<br>

<center><img src = '/post_img/200108/image6.png' width="600"/></center>

이는 또한 Training set($X_{train}$)의 분포와 Test set($X_{test}$)의 분포에 대해 차이를 발생시킨다. 이러한 현상을 Covariate Shift라고 부른다.






$$ \hat x ^{(k)}  = {x^{(k)}-E[x^{(k)}] \over \sqrt{Var[x^{(k)}]}} $$



<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
