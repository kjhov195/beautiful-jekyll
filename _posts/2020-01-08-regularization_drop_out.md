---
layout: post
title: Regularization(1)-dropout
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>
### Overfitting

<br>

<center><img src = '/post_img/200108/image1.png' width="600"/></center>

오버피팅(overfitting)이란 __training data__ 를 과하게 학습하는 것을 뜻한다. training data를 과하게 학습하는 것이 어떤 문제가 될까?

일반적으로 Training data는 실제 데이타의 부분 집합이므로 __training data__ 에 대해서는 __Error가 감소__ 하지만, __Test data__ 에 대해서는 __Error가 증가__ 하게 된다.

<br>

<center><img src = '/post_img/200108/image2.png' width="600"/></center>

위 그림의 overfitting된 모델의 예시에서도, Test dataset에 대하여 Error가 꽤 증가하는 것을 확인할 수 있다.

<br>

<center><img src = '/post_img/200108/image3.png' width="600"/></center>

과적합을 할수록 Training dataset에 대한 Error는 계속 줄어들겠지만, Validation set에 대한 Error는 어느 순간부터 증가하기 마련이다. 이 때 우리는 모델이 __Overfitting__ 되어 있다고 한다.


<br>
<br>
### Regularization

Overfitting을 해결하기 위해 Regularization을 도입하게 되며, Deep learning에서 많이 사용되는 Regularization 방법에는 다음과 같은 것들이 있다.



<br>
<br>
### Dropout

<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
