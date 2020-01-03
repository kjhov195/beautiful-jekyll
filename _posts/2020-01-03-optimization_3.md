---
layout: post
title: Optimization(3)-AdaGrad/RMS prop/Adam
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는
Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.

<br>
<br>
### AdaGrad

AdaGrad는 SGD를 개선한 형태의 optimizer이다.

이전 iteration에서의 weight에서 learning rate $\cdot$ gradient를 빼주는 형태를 가진 SGD에서 grad_squared term을 반영해준 것인데, pseudo-code를 통해 살펴보면 이해하기가 쉽다.

```
#pseudo-code for AdaGrad
grad_squared = 0
while True:
  dw = compute_gradient(w)
  grad_squared += dw^2
  w += -learning_rate*dw/(sqrt(grad_squared)+1e-7)
```

<br>
<br>
### RMS prop

RMS prop은 AdaGrad를 다시 개선한 형태의 optimizer이다.

```
#pseudo-code for RMS prop
grad_squared = 0
while True:
  dw = compute_gradient(w)
  grad_squared += decay_rate*grad_squared+(1-decay_rate)*dw^2
  w += -learning_rate*dw/(sqrt(grad_squared)+1e-7)
```

<br>
<br>
### Adam

Adam은 앞서 살펴본 Momentum/AdaGrad/RMS prop의 장점만을 모아 하나로 결합한 형태의 optimizer이며, 거의 대부분의 optimization 문제에서 상당히 좋은 default optimizer로 사용할 수 있다.

```
#pseudo-code for Adam
first_moment = 0
second_moment = 0
while True:
  dw = compute_gradient(w)
  first_moment  = beta1*first_moment +(1-beta1)*dw
  second_moment = beta2*second_moment+(1-beta2)*dw^2
  first_unbias  = first_moment /(1-beta1)^t
  second_unbias = second_moment/(1-beta2)^t

  w += -learning_rate*first_moment/(sqrt(second_moment)+1e-7)
```

일반적으로 $\beta_1$, $\beta_2$, $\text{learning_rate}$은 다음을 기본적으로 사용한다.

$$
\begin{align*}
\beta_1 &= 0.9\\
\beta_2 &= 0.999\\
\text{learning rate} &= 1e-3\;\;or\;\;5e-4\\
\end{align*}
$$


<br>
<br>
### Reference

[CS231n: Lecture 7, Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7)
