---
layout: post
title: CNN(Convolutional Neural Networks)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>
### Convolutional Neural Networks

앞서 우리는 [Softmax Classifier](https://kjhov195.github.io/2020-01-05-softmax_classifier_3/), [Neural Networks(MLP)](https://kjhov195.github.io/2020-01-07-weight_initialization/) 등의 모델을 통하여 MNIST dataset의 classification 문제를 풀어보았다. 지금까지는 $28 \times 28$ 크기의 이미지를 다루기 위하여 $28 \times 28$ 행렬을 $1 \times 784$ 형태의 벡터로 reshape 해주어 input으로 사용했다. 하지만 이러한 방식은 이미지 데이터의 __공간 정보__ 유실을 불러일으키고, 모델의 성능을 저하시킬 수 밖에 없다.

반면, CNN에서는 __Convolution layer__ 을 도입하여 이 문제를 해결하게 된다. CNN은 $28 \times 28$을 일렬로 펼친 형태의 벡터로 reshape해주지 않고, $28 \times 28$의 이미지 데이터를 그대로 사용한다.

그렇다면 __Convolution layer__ 가 무엇인지에 대하여 조금 더 자세히 살펴보도록 하자.

<br>
<br>
### Convolution Layer
<br>

<center><img src = '/post_img/200110/image1.png' width="700"/></center>



<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

[vdumoulin github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

[Pytorch Documentation](https://pytorch.org/docs/stable/nn.html#conv2d)
