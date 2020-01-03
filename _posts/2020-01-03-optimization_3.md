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

이전 iteration에서의 weight에서 learning rate $\cdot$ gradient를 빼주는 형태를 가진 SGD에서 grad_squared term을 반영해준 것인데, pseudo-code를 통해 살펴보면 조금 더 이해하기가 쉽다.

```
#pseudo-code for AdaGrad
grad_squared = 0
while True:
  dw = compute_gradient(w)
  grad_squared += dw^2
  w += -learning_rate*dw/(sqrt(grad_squared)+1e-7)
```

즉, AdaGrad는 각 dimension마다 gradient의 historical sum of squares로 element-wise scaling을 해주는 term을 추가한 것이다.(scaling term에 1e-7이 더해지는 이유는 우리가 나누는 값(scaling term)이 0이 되지 않도록 해주기 위함이다.)

<br>

<center><img src = '/post_img/200103/image1.png' width="450"/></center>

이러한 scaling term을 통하여 항상 gradient가 큰 coordinate(위 그림에서 수직축에 해당)의 경우, scaling term이 클 것이다. scaling term은 나눠지는 값이므로, 결과적으로 우리는 해당 dimension에 대한 progress를 늦출 수 있게 된다. 반면, 항상 gradient가 작은 coordinate(위 그림에서 수평축에 해당)의 경우, scaling term이 작을 것이다. 역시 나눠지는 값이므로, 결과적으로 우리는 해당 dimension에 대한 progress를 촉진(Accelerate)시킬 수 있게 된다.

또한, gradient가 점점 작아지는 Convex case의 경우 AdaGrad의 아이디어는 상당히 좋은 성능을 보인다. 왜냐하면 Minimum으로 갈 수록 gradient가 줄어들 것이고, 이 때문에 매 step마다의 converge의 속도를 늦출 수 있게 되어 결과적으로는 조금 더 빠르고, 정확한 converge가 가능하도록 해준다.

<br>

<center><img src = '/post_img/200103/image1.png' width="450"/></center>

반면 Non-convex case의 경우 조금 문제가 된다. 예를들어 Saddle point의 경우를 생각해보자. Saddle point 근방에 들어올 경우, gradient가 매우 작으므로 해당 point 근방을 벗어나기가 어려워지고, 더 이상 학습이 일어나지 않게 된다는 문제점이 존재한다.


<br>
<br>
### RMS prop
RMS prop은 AdaGrad를 다시 개선한 형태의 optimizer이며, Decay rate라는 개념을 도입한 AdaGrad optimizer라고 볼 수 있다. Decay rate은 하나의 hyper-prameter로써 보통 0.9, 0.99를 많이 사용한다.

Gradient 제곱의 합을 구하는 것은 AdaGrad와 동일한데, Decay rate을 도입함으로써 iteration을 거치는 과정에서 scaling term이 서서히 줄어들게(leaking) 해준다. AdaGrad의 장점은 그대로 유지하면서, Step-size가 0이되어 학습이 일어나지 않게되는 문제를 해결한 것이다.

```
#pseudo-code for RMS prop
grad_squared = 0
while True:
  dw = compute_gradient(w)
  grad_squared += decay_rate*grad_squared+(1-decay_rate)*dw^2
  w += -learning_rate*dw/(sqrt(grad_squared)+1e-7)
```

<br>

<center><img src = '/post_img/200103/image1.png' width="450"/></center>

RMS prop은 학습이 조기에 종료될 수 있는 AdaGrad보다 훨씬 더 안정적이고, 빠른 속도로 학습이 끝까지 진행될 수 있도록 해준다.


<br>
<br>
### Adam

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

Adam은 앞서 살펴본 Momentum/AdaGrad/RMS prop의 장점만을 모아 하나로 결합한 형태의 optimizer이다. "```first_moment```" 의 경우 momentum의 개념을 도입한 부분이고, "```second_moment```" 의 경우 AdaGrad의 개념을 도입한 부분이다. 최종적으로 w를 update해주는 "```w += ```"부분의 경우, RMS prop과 상당히 유사한 형태인 것을 확인할 수 있다.

일반적으로 $\beta_1$, $\beta_2$, $\text{learning_rate}$은 기본적으로 다음 값들을 사용한다.

$$
\begin{align*}
\beta_1 &= 0.9\\
\beta_2 &= 0.999\\
\text{learning rate} &= 1e-3\;\;or\;\;5e-4\\
\end{align*}
$$

<br>

<center><img src = '/post_img/200103/image1.png' width="450"/></center>

Adam optimizer의 경우 거의 대부분의 optimization 문제에서 꽤나 좋은 성능을 보이며, 그러한 이유로 많은 문제/영역에서 default optimizer로 사용되고 있다.


<br>
<br>
### Reference

[CS231n: Lecture 7, Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7)
