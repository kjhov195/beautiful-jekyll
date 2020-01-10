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

앞서 우리는 [Softmax Classifier](https://kjhov195.github.io/2020-01-05-softmax_classifier_3/), [Neural Networks(MLP)](https://kjhov195.github.io/2020-01-07-weight_initialization/) 등의 모델을 통하여 MNIST dataset의 classification 문제를 풀어보았다. 지금까지는 $28 \times 28$ 크기의 이미지를 다루기 위하여 $28 \times 28$ 행렬을 $1 \times 784$ 형태의 벡터로 reshape 해주어 input으로 사용했다. 하지만 이러한 방식은 데이터의 __공간 정보__ 를 유실시키는 결과를 가져오고, 이로인하여 모델의 성능은 저하될 수밖에 없다.

반면, CNN에서는 __Convolution layer__ 을 도입하여 이 문제를 해결하게 된다. CNN은 $28 \times 28$을 일렬로 펼친 형태의 벡터로 reshape해주지 않고, $28 \times 28$의 이미지 데이터를 그대로 사용한다.

그렇다면 __Convolution layer__ 가 무엇인지에 대하여 조금 더 자세히 살펴보도록 하자.

<br>
<br>
### Convolution Layer

<br>

<center><img src = '/post_img/200110/no_padding_no_strides.gif' width="300"/></center>

[vdumoulin github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)에 Convolution layer의 연산 방식을 잘 나타낸 그림이 있어서 가져와 보았다.

위 그림은 $4 \times 4$의 input data에 $3 \times 3$의 filter를 적용하여 $2 \times 2$의 output을 구하는 과정을 보여준다. output 행렬의 각 element를 계산하는 방식은 아래 예시에서 조금 더 자세히 살펴보도록 하겠다.

<br>

<center><img src = '/post_img/200110/image1.png' width="700"/></center>

간단한 설명을 위하여 $5 \times 5$의 input data가 있다고 하자.(초록색)

Convolution layer는 filter 행렬을 사용하여 연산하고(주황색), output 행렬을 구해낸다.(분홍색)

<br>

<center><img src = '/post_img/200110/image0.png' width="700"/></center>

$$
\begin{align*}
1*1 + 2*0 + 3*1\\
+0*0+1*1+5*0\\
+1*1+0*0+2*1\\
= 8
\end{align*}
$$


output 행렬의 1행 1열의 값을 구하는 과정은 위와 같이 element-wise multiplication 연산을 계산하여 최종적으로 합해준다. 나머지 elements에 대해서도 똑같은 방식으로 연산해주면 최종적인 output 행렬을 구할 수 있다.

이러한 방식으로 Filter를 사용하여 연산해주는 layer를 Convolution Layer라고 한다.

이번에는 또 다른 예시로 MNIST 데이터를 통해 실제 데이터에서 연산되는 과정을 한 번 살펴보도록 하자.

<br>

<center><img src = '/post_img/200110/image2.png' width="700"/></center>

$28 \times 28$의 input data에 $3 \times 3$의 filter를 적용하여 계산하여 output 행렬의 1행 1열의 성분을 계산하는 과정을 나타낸 것이다. 마찬가지로 element-wise multiplication을 해준 후, 이를 더해주는 과정을 거쳐 계산이 이루어진다.

<br>
<br>
### ```torch.nn.Conv2d()```

<br>

<center><img src = '/post_img/200110/image4.png' width="700"/></center>

Pytorch Pytorch 공식 홈페이지의 [Documentation](https://pytorch.org/docs/stable/nn.html#conv2d)에 나와있는 Convolution layer를 구현한 함수이다. 여러 옵션을 줄 수 있는데, 이 중에서 __stride__ 와 __padding__ 에 대하여 자세히 살펴보도록 하겠다.

<br>
<br>
### stride

<br>

<center><img src = '/post_img/200110/no_padding_strides.gif' width="300"/></center>



<br>
<br>
### zero padding

<br>

<center><img src = '/post_img/200110/full_padding_no_strides.gif' width="300"/></center>





<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

[vdumoulin github](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

[Pytorch Documentation](https://pytorch.org/docs/stable/nn.html#conv2d)
