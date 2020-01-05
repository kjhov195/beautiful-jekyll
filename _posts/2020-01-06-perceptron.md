---
layout: post
title: Perceptron
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.

<br>
<br>
### Perceptron

Perceptron은 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘이이며, Frank Rosenblatt가 1957년에 제안한 초기 형태의 인공 신경망이다.

<br>

<center><img src = '/post_img/200106/image4.png' width="600"/></center>

퍼셉트론은 실제 뇌를 구성하는 신경 세포 뉴런의 동작과 유사한데, 신경 세포 뉴런의 그림을 먼저 보도록 하자. 뉴런은 가지돌기에서 신호를 받아들이고, 이 신호가 일정치 이상의 크기를 가지면 축삭돌기를 통해서 신호를 전달한다.

<br>

<center><img src = '/post_img/200106/image5.png' width="300"/></center>

퍼셉트론의 구조도 이와 비슷하다. 그림에서의 원은 뉴런, $W$는 신호를 전달하는 축삭돌기의 역할을 하게 된다. 각 뉴런에서 보내지는 입력값 $x$를 가중치 $W$에 곱해주고, 이 값을 Activation 함수를 통과시켜 뉴런 $y$로 전달해주는 것이다.

그런데 이 구조를 어디서 많이 보지 않았는가? 이 Activation 함수에 $Sigmoid$ 함수를 사용하면 Logistic Regression이 되고, $Softmax$ 함수를 사용하면 Softmax classifier(Multiclass Logistic Regression) 모형이 된다. 즉, 하나의 로지스틱 회귀(혹은 Softmax Classifier) 모형은 Perceptron의 special case라고 볼 수 있는 것이다.

이러한 하나의 perceptron은 Neural Networks의 전체적인 구조 안에서 뉴런 역할을 하게 된다.

<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

[PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/60680)
