---
layout: post
title: Activation fucntion-Sigmoid/tanh/ReLU/LeakyReLU/Maxout/ELU
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>
### Activation function

<br>

<center><img src = '/post_img/200107/image1.png' width="450"/></center>

일반적인 Neural Networks는 $\sum_i W_iX_i$를 계산한 후에 Sigmoid와 같은 함수 $f$를 통하여 transformation해주는 작업을 각 layer마다 반복하게 된다. 이러한 transformation function $f$를 우리는 __Activation Function__ 이라고 한다.

이번 포스트에서는 Neural Networks에서 Activation function으로 Sigmoid function을 사용하였을 때의 단점을 살펴보고, 이를 보완하여 고안된 다양한 Activation function을 살펴보도록 하겠다.

<br>
<br>
### Sigmoid

$$ \sigma(x) = \frac 1 {1+e^{-x}}$$

앞서 우리는 [여기](https://kjhov195.github.io/2020-01-05-softmax_classifier_3/)에서 Sigmoid 함수를 Activation함수로 사용한 Softmax classifier를 통하여 MNIST dataset classification을 해보았다.

<br>

<center><img src = '/post_img/200107/image2.png' width="450"/></center>

Sigmoid 함수의 경우 0부터 1까지 범위의 값을 가지며, 통계학에서 아주 많이 사용되는 함수이다.

하지만 Sigmoid 함수의 경우 Neural Networks에서 Activation 함수로 사용하기에는 __매우__ 부적절한 함수이며, 크게 3가지 이유가 존재한다.

(1) Vanishing Gradient

<br>

<center><img src = '/post_img/200107/image3.png' width="450"/></center>

위와 같은 Computational graph를 생각해보자. 우리는 최종적으로 $\partial L \over \partial x$를 계산해야 하며, 이는 Back propagation을 통하여 ${\partial L \over \partial x} = {\partial L \over \partial \sigma} \cdot {\partial \sigma \over \partial x}$와 같이 계산할 수 있다.

여기서 문제가 되는 부분은 Sigmoid 함수에 대한 미분 값인 ${\partial \sigma \over \partial x}$의 크기에 대한 문제이다. 다음 그림을 살펴보자.

<br>

<center><img src = '/post_img/200107/image4.png' width="450"/></center>

위 그림에서 빨간 박스에 해당하는 부분은 Gradient가 거의 0에 가까운 아주 작은 숫자를 가진다. 즉, $x$가 0보다 꽤 작거나, 클 경우 ${\partial \sigma \over \partial x} \approx 0$가 된다.

<br>

<center><img src = '/post_img/200107/image5.png' width="450"/></center>

이는 Multi layer perceptron에서 큰 문제가 된다. layer가 많을 경우 최종적인 Gradient를 구하기 위하여 local Gradient를 상당히 많이 곱해주게 되는데(Back propagation), 이 과정에서 0에 가까운 값들을 계속 곱해주게 되는 것이다. 이로 인하여 앞단에 위치한 layer일 수록 Gradient 값들을 제대로 구하지 못하고 거의 0에 수렴해버리는 문제가 발생하며, 이로 인하여 Weight들이 제대로 update되지 못하게 된다. 이러한 현상을 __Vanishing Gradient__ 라고 한다.

<br>

(2) Not zero-centered

Sigmoid 함수의 경우 output 값이 0.5를 중심으로 하며, 0과 1사이에 위치해 있다. 하지만 Neural Networks의 경우 이러한 구조의 Activation 함수는 좋은 성능을 보이지 못한다.

앞서 Vanishing Gradient에서 살펴보았던 그림을 다시 살펴보자.

<br>

<center><img src = '/post_img/200107/image3.png' width="450"/></center>

여기서 Activation function(sigmoid)의 결과 값 $\sigma(\sum_i w_i x_i + b)$를 각 $w_i$로 미분한 값들을 구해보자. 즉, $\frac {\partial \sigma} {\partial w_1}$, $\frac {\partial \sigma} {\partial w_2}$, $\cdots$, $\frac {\partial \sigma} {\partial w_i}$, $\cdots$
에 대해 생각해보자. 이 값들은 다음과 같이 구할 수 있다.

$$
\begin{align*}
\frac {\partial \sigma} {\partial w_1} &= x_1 > 0\\
\frac {\partial \sigma} {\partial w_2} &= x_2 > 0\\
\vdots \;\;\; &= \; \vdots \\
\frac {\partial \sigma} {\partial w_i} &= x_i > 0\\
\vdots \;\;\; &= \; \vdots \\
\end{align*}
$$

multi layer를 가정할 경우, 이 그림의 앞단에서 또 다른 sigmoid 값을 input $x$로 받았다고 생각할 수 있다. Sigmoid 함수의 결과 값은 항상 양수이므로, 이 경우 $x$는 항상 양수인 것이다.

한편, 최종 out


<br>
<br>
### tanh

<br>
<br>
### ReLU

<br>
<br>
### LeakyReLU

<br>
<br>
### Maxout


<br>
<br>
### ELU


<br>
<br>
### Example

<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
