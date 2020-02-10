---
layout: post
title: image dataset(1)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>


이번 포스트와 다음 포스트에서는 Computer Vision 분야에서 많이 사용되는 대표적인 데이터셋을 살펴보도록 하겠다. 이번 포스트에서는 Image Classification(Image Recognition) 문제에서 주로 사용되는 데이터셋들을 위주로 살펴보도록 하겠다.

<br>
<br>
### MNIST

$
\begin{align*}
\text{Num of classes: }&10\\
\text{Size of images: }&28 \times 28\\
\text{Training set: } &60000 \times 1 \times 28 \times 28\\
\text{Test set: }&10000  \times 1 \times 28 \times 28\\
\end{align*}
$

MNIST 데이터셋은 손으로 쓴 숫자 이미지로 이루어진 대형 데이터셋이며, 60,000개의 Training dataset과 10,000개의 Test dataset으로 이루어져 있다.

MNIST 데이터셋(Modified National Institute of Standards and Technology database)은 NIST의 오리지널 데이터셋의 샘플을 재가공하여 만들어졌다.

MNIST의 Training set의 절반과 Test set의 절반은 NIST의 Training set에서 취합하였으며, 그 밖의 Training set의 절반과 Test set의 절반은 NIST의 Test set으로부터 취합되었다.


각 데이터는 0에서 9까지의 자연수 중 하나에 대응되는 숫자에 대한 데이터이며, 1×28×28, 총 784개의 픽셀의 색에 대한 정보를 담은 행렬이다. 각 픽셀에 대응되는 행렬의 성분은 0부터 255사이의 숫자를 가지고 있는데, 까만색에 가까운 픽셀일수록 0에 가까운 값을, 흰색에 가까운 픽셀일수록 255에 가까운 값을 가지게 된다.


<br>
<br>
### SVHN


<br>
<br>
### STL10




<br>
<br>
### CIFAR-10
$
\begin{align*}
\text{Num of classes: }&10\\
\text{Size of images: }&32 \times 32\\
\text{Training set: } &50000 \times 3 \times 32 \times 32\\
\text{Test set: }&10000  \times 3 \times 32 \times 32\\
\end{align*}
$

CIFAR는 Canadian Institute For Advanced Research의 줄임말이다.

CIFAR-10과 CIFAR-100은 80,000,000만개의 [tiny images dataset](http://people.csail.mit.edu/torralba/tinyimages/)의 subset-data로써, Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton에 의해 수집된 데이터셋이다.

이미지의 크기는 $32 \times 32$이며, 3가지 색상 channel로 이루어져 있고, training set의 경우 50,000개, test set의 경우 10,000개의 데이터로 이루어져 있다.

<br>



Target의 경우 위와 같이 Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck의 10개의 class로 구성되어 있으며, 각 class는 서로 완벽하게 차별(completely mutually exclusive)되는 데이터셋이다. 즉, 어떤 데이터이든 10개의 class중 완벽하게 하나의 class에만 속하도록 구성되어 있다.

<br>
<br>
### CIFAR100
$
\begin{align*}
\text{Num of classes: }&100\\
\text{Size of images: }&32 \times 32\\
\text{Training set: } &50000 \times 3 \times 32 \times 32\\
\text{Test set: }&10000  \times 3 \times 32 \times 32\\
\end{align*}
$

CIFAR-100의 경우, 20개의 Supeerclass와 100개의 class로 구성된 데이터셋이며, 데이터의 수와 사이즈는 CIFAR-10과 동일하다.

<br>

Target의 경우 위와 같이 100개의 Classes로 구성되어 있으며, 그와 동시에 20개의 Superclass로 나눌 수도 있다.

_사실, 이 데이터셋에는 조그마한 오류가 있는데 버섯(mushrooms)의 경우 과일/야채(fruit or vegetables)가 아니고, 곰(bears)의 경우 육식동물이 아니다._ :)


<br>
<br>
### ImageNet
$
\begin{align*}
\text{Num of classes: }&21841\\
\text{Num of data: }&14197122\\
\end{align*}
$

Stanford University의 Li Fei-Fei 교수를 중심으로 데이터베이스가 제작되었으며, Google, Microsoft와 같은 기업 또한 함께 참여하여 제작한 데이터셋이다.

ImageNet Dataset의 경우, __ILSVRC: ImageNet Large Scale Visual Recognition Competition__ 에서 사용되는 것으로 유명하다.

총 21,841개 classes의 14,197,122개 데이터로 구성되어 있으며, 다양한 해상도의 image size로 구성되어 있다. 데이터셋 구성 내역에 대한 자세한 설명은 [이곳](http://image-net.org/about-stats)의 설명을 참고하면 된다.

보통 $264 \times 264$ pixels로 crop된 sub-sampled images를 사용하는 것이 일반적이다.

<br>
<br>
### COCO

$
\begin{align*}
\text{Num of classes: }&80\\
\text{Num of data: }& \text{2.5 million labeled instances in 328k images}\\
\end{align*}
$

Microsoft의 COCO Dataset은 약 33만개의 데이터로 구성되어 있으며, 여러 버전에 걸쳐 공개되었다. 각 데이터셋의 규모는 다음과 같다.

train2017 : 118,287
annotation (train2017) : 117,266
val2017 : 5,000
annotation (val2017) : 4,952

MS COCO paper에는 91개의 class라고 명시되어 있지만, 실제로는 80개의 class가 제공된다.




<br>
<br>
### Reference

https://www.cs.toronto.edu/~kriz/cifar.html

https://laonple.blog.me/220643128255


http://image-net.org/about-stats

https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
