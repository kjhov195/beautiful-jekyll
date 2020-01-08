---
layout: post
title: Activation fucntion-tanh/ReLU/LeakyReLU/Maxout/ELU
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>

### Sigmoid

<br>

<center><img src = '/post_img/200107/image2.png' width="450"/></center>


[앞선 포스트](https://kjhov195.github.io/2020-01-07-activation_function_1/)에서 Activation 함수로써의 Sigmoid 함수에 대하여 세 가지 문제점을 살펴보았다. 그 중 __Vanishing Gradient__ 와 __Not zero centered__ 는 Neural Networks의 성능 저하에 큰 영향을 미치게 된다.


<br>

<center><img src = '/post_img/200107/image100.png' width="800"/></center>

이러한 문제점을 해결하기 위하여 다양한 Activation 함수가 고안되었고, 오늘은 tanh/ReLU/LeakyReLU/Maxout/ELU 함수에 대해 살펴보고자 한다.



<br>
<br>
### tanh

<br>

<center><img src = '/post_img/200107/image7.png' width="450"/></center>


<br>
<br>
### ReLU

<br>

<center><img src = '/post_img/200107/image8.png' width="450"/></center>


<br>
<br>
### LeakyReLU

<br>

<center><img src = '/post_img/200107/image9.png' width="450"/></center>


<br>
<br>
### Maxout


<br>
<br>
### ELU

<br>

<center><img src = '/post_img/200107/image10.png' width="600"/></center>


<br>
<br>
### Example

```
example
```


<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
