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
### Covariate Shift


<br>

<center><img src = '/post_img/200108/image6.png' width="600"/></center>

이는 또한 Training set($X_{train}$)의 분포와 Test set($X_{test}$)의 분포의 차이가 있는 경우, 이를 __Covariate Shift__ 라고 부른다.

<br>

<center><img src = '/post_img/200108/image8.png' width="600"/></center>

Covariate Shift는 모델의 성능 저하에 큰 영향을 미친다. 그 이유를 직관적으로 잘 설명해주는 자료가 있어 [JUNSIK HWANG님의 블로그](https://jsideas.net/batch_normalization/)에서 위 그림을 가지고 왔다.

고양이와 강아지를 분류하는 문제를 풀고있으며, Training dataset에는 러시안 블루 고양이만 있고, Test dataset에는 페르시안 고양이만 있다.(즉, Covariate Shift를 일부러 만들어보자.)

이 때 Training data에 있는 러시안 블루 고양이에 대한 우리가 적합시킨 모델의 분류 정확도는 99%에 달한다.

하지만 Test dataset에는 페르시안 고양이만 있는 상황이다. 이 때 우리가 train시킨 모델에 이러한 Test dataset을 적용하면 어떤 결과가 발생할까?

페르시안 고양이의 털 색깔(흰색)을 보고 _"Training set에서는 회색 털을 가지고 있어야 고양이라고 배웠는데, Testset의 이 친구는 하얀색 털을 가지고 있구나. 그럼 이 친구는 강아지 일 수도 있겠다."_ 라는 판단을 할 수 있게 되고, 결과적으로 오분류의 가능성이 높아진다.

즉, Training dataset과 Input dataset의 분포에 대한 차이는 모델의 성능 저하에 큰 영향을 미칠 수 있는 것이다.


<br>
<br>
### Internal Covariate Shift

Neural Networks에서 모든 Training data를 한 번에 사용하지 않고 Mini batch를 사용할 경우, 각 step에서 사용되는 Training data는 매번 달라지게 된다. 이렇게 배치 간의 데이터 분포가 다른 경우를 __Internal Covariate Shift__ 라고 한다.

이러한 __Internal Covariate Shift__ 문제는 Layer의 수가 많으면 더욱 더 큰 문제가 된다.

일반적인 Neural Networks에서는 여러 layers를 사용하며, 각 layer마다 input을 받아 linear combination을 구한 후 Activation function을 적용하여 output을 구해주는 작업이 이루어 진다.

<br>

<center><img src = '/post_img/200108/image7.png' width="600"/></center>

결과적으로 이 때문에 각 layer의 input data $x$의 분포(Distribution)가 달라지게 되며, 뒷단에 위치한 layer일 수록 변형이 누적되어 input data의 분포는 상당히 많이 달라지게 된다.

이런 상황이 발생할 경우, 모델이 일관적인 학습을 하기가 어려워진다.



<br>
<br>
### Batch Normalization

Batch Normalization은 loffe and Szegedy(2015)에 의하여 제시된 아이디어이다.






$$ \hat x ^{(k)}  = {x^{(k)}-E[x^{(k)}] \over \sqrt{Var[x^{(k)}]}} $$



<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
