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
### XOR problem

<br>
##### XOR gate

앞의 포스트에서 XOR gate는 단층 perceptron으로 풀 수 없다는 사실을 확인하였다.

XOR gate가 무엇인지 다시 한 번 살펴보자.

<br>

|  <center> $x_1$ </center> |  <center> $x_2$</center> | <center> $y$</center> |  
|:--------|:--------:|--------:|--------:|
| <center>  0 </center> | <center> 0 </center> | <center> 0 </center> |
| <center>  0 </center> | <center> 1 </center> | <center> 1 </center> |
| <center>  1 </center> | <center> 0 </center> | <center> 1 </center> |
| <center>  1 </center> | <center> 1 </center> | <center> 0 </center> |

XOR 게이트는 두 개의 입력 값 $x_1$과 $x_2$가 모두 0이거나, 모두 1인 경우에만 출력값 $y$가 0이고, 나머지의 경우에는 1이 나오는 구조를 가지고 있다. 이를 그림으로 나타내면 다음과 같다.

<br>

<center><img src = '/post_img/200106/image3.png' width="450"/></center>



<br>
<br>
### Multi layer perceptron

1969년, MIT AI laboratory의 창시자인 Marvin Minsky교수는 single layer perceptron으로는 XOR 문제를 해결할 수 없으며, XOR 문제를 풀기 위해서는 Multi layer perceptron을 도입이 필요하다는 것을 밝혔다.

단층 perceptron으로는 XOR 문제를 풀 수 없지만 Layer의 수를 더 늘릴 경우 XOR 문제를 풀 수 있게 되는 것이다.

이러한 구조의 perceptron을 MLP(Multi Layer Perceptron)이라고 한다. MLP란 input layer와 output layer 사이에 hidden layers가 추가된 구조의 perceptrons를 의미한다.(참고로 하나의 hidden layer를 사용하는 MLP의 경우 Vanilla Neural Networks라고도 부르기도 한다.)

하지만 1969년 당시에는 이러한 Multi layer perceptron을 학습할 수 있는 방법을 발견하지 못했지만, 이후에 Back propagation이 고안되면서 Multi layer perceptron을 적합시킬 수 있게 된다. Back propagation에 대한 자세한 내용은 다음 포스트에서 살펴보도록 하겠다.

<br>
<br>
### 작성 중...

작성 중...

<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

[PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/60680)
