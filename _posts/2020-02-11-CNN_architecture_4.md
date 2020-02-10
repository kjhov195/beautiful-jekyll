---
layout: post
title: CNN Architecture(4)-ResNet
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

<br>
<br>
### ILSVRC'15

<br>

이번 포스트에서 살펴볼 모델은 ILSVRC'15의 Classification task에서 1등을 차지한 ResNet이다.

<br>
<br>
### ResNet(2015)

paper: [Deep Residual Learning for Image Recognition, He et al., 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

기본적으로 Layer의 수가 많아지면 더 좋은 성능을 가져와야 할 것 같지만, ResNet이 아닌 다른 모델들의 경우 Layer가 많아짐에 따라 Error rate가 커지는 결과를 보였다. 하지만 ResNet의 경우 이러한 문제를 해결하게 되고, ILSVRC'14까지 선보였던 여느 다른 모델들보다 기하급수적으로 많은 수의 Layer를 사용하여 3.6% top5 error rate를 달성하게 된다.

ResNet의 경우 ImageNet data를 training하는데 8개의 GPU를 사용하였을 때 약 2~3주가 걸릴 정도로 오래 걸린다. 다만 test time 때에는 VGGNet보다 더 빠른 속도를 가진다.

ResNet의 경우 초기에 $224 \times 224$의 image를 $56 \times 56$의 크기로 줄여준 뒤에 이를 유지하며, skip connection을 도입하여 효율적인 training을 가능하게 한다.


이 포스트는 현재 작성 중에 있습니다.

이 포스트는 현재 작성 중에 있습니다.

<br>
<br>
