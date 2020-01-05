---
layout: post
title: Multi layer perceptron(MLP)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.

<br>
<br>
### Multi layer perceptron

하지만, 단층 perceptron으로는 이러한 XOR 문제를 풀 수 없지만 Layer의 수를 더 늘릴 경우 XOR 문제를 풀 수 있다는 사실을 알게된다.

이러한 구조의 perceptron을 MLP(Multi Layer Perceptron)이라고 한다. MLP란 input layer와 output layer 사이에 hidden layers가 추가된 구조의 perceptrons를 의미한다.

참고로 하나의 hidden layer를 사용하는 MLP의 경우 Vanilla Neural Networks라고도 부르기도 한다.


<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

[PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/60680)
