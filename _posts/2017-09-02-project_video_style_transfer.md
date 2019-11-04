---
layout: post
title: Animation Generation Project
subtitle: Video Style Transfer, Edge Detection
category: Project
use_math: true
---

## 2019 Spring, YBIGTA Conference 1st
2019 YBIGTA Conference에서 우리 팀이 1등을 수상하였다. 우리 팀의 프로젝트 주제는 현실 영상을 애니메이션 스타일로 변환해주는 모델을 만들어보는 것이었다.

<img src = '/post_img/190902/style_transfer_1.png'/>

_Loving Vincent(2017)_

2017년에 개봉하여 흥행에 성공했었던 _Loving Vincent(2017)_ 에서 아이디어를 착안하여 팀원들과 이 프로젝트를 시작하게 되었다. 러빙빈센트의 경우 107명의 화가가 약 2년 동안 65,000프레임에 달하는 양을 작화했다. 실제 영화를 찍은 뒤, 이를 그림으로 그려내는 작업에 아주 많은 시간과 자원, 노력이 들어간 것이다. 만약 Style Transfer Model이 잘 작동하여 이를 수행해준다면, 혹은 그림으로 그려내는 작업의 밑바탕이 되어준다면 좋지 않을까라는 생각이었다.


<br>
<br>
## Style Transfer

기본적인 Style Transfer 기본적인 Idea를 살펴보자.

<img src = '/post_img/190902/style_transfer_2.png'/>

_Style Transfer_

Style을 적용하고자 하는 content image와 style image를 네트워크에 통과시킬 때 Feature Map을 얻을 수 있다. 이 각각의 feature map을 바탕으로 새롭게 합성될 영상의 feature map이 비슷하도록 최적화하는 과정을 _Style Transfer_ 라고 할 수 있다.

<br>
<br>
## Model searching

사실 작년에 KT ATC에서 다양한 프로젝트를 진행하면서 Image에 대한 Style Transfer Model(Vanilla gan, DCGAN, Cycle GAN 등)은 많이 접해보았지만, Video의 Style Transfer는 처음 접해보는 영역이었다.

처음 떠오른 모델은 GAN이였고, 역시 최종적으로 사용한 모델 또한 GAN이었다. 하지만 GAN의 치명적인 단점인 _동일한 image를 input으로 주어도 styled된 결과가 매번 달라지는 단점_ 때문에 조금 더 안정적인 모델이 있지 않을까 하는 생각에 다른 model 또한 searching해 보았다.

모델 Searching 과정에서 NVIDIA & MIT의 Video-to-Video Synthesis 등의 모델을 찾았지만 Paired Dataset에 대해서만 학습이 가능하다는 한계가 있었다. 우리 팀이 직접 Paired Dataset을 만들 수는 없는 상황이었으므로 안타깝게도 이 모델은 사용할 수 없었다. GAN based model이 아닌 다른 여러 Style Transfer 모델들 또한 Video가 아닌 Image에 대한 연구가 대부분이었다.

결국 다시 처음으로 돌아가서 기본 Base가 되는 모델로 다음 두 모델을 선택하게 되었다.


<br>
<br>
## 1. Artistic Style Transfer for Videos

우리가 선택한 첫 번째 모델은 Artistic Style Transfer for Videos(2016, Cornell University)이다.

<img src = '/post_img/190902/style_transfer_3.png'/>

_Artistic Style Transfer for Videos(2016, Cornell University)_

이 논문에 대한 간단한 설명을 덧붙이면 다음과 같다.

- Image style transfer를 video sequence style transfer로 확장하였음.
- artistic image를 특정 style로 변환하여 영상 전체에 painting.
- Transfer를 정규화하고, 프레임 간의 자연스러운 전환을 위하여 두 프레임 사이의 편차를 penalize하는 temporal constraint.

논문을 읽었을 때 우리 프로젝트의 목적과 잘 부합해 보이는 모델이라는 생각이 들었다. 하지만 문제는...

---

#### Lua

이 논문의 저자가 git에 paper code를 공개해놓았는데, Lua로 만들어진 모델이었다.(https://github.com/jcjohnson/neural-style)

<img src = '/post_img/190902/style_transfer_4.png' width="250"/>

_Lua_

tf, Pytorch만 접해왔던 나에게는 큰 결심이 필요한 도전이었다. 간단하게나마 Lua를 읽을 수 있기 위해 며칠을 Lua만 쳐다봤는지 모르겠다. 게다가 대학원에 Lua를 써본 사람이 없어서, 환경 구축하는 것도 너무너무 힘들었고 오래 걸렸다. :sob::sob::sob:

---

일주일 정도 매달려서 결국 구현에 성공했다. 물론 Lua에 익숙하지 않아 기본적인 setting에서 모델을 사용했지만, Image를 Input으로 사용했을 때에는 꽤나 성능이 잘나왔다. 우리 연구실의 대표 미남 연구원을 Input Image로 사용하여 _van gogh_ 의 _starry night_ 의 Style로 Image Style Transfer를 해보았다.

<img src = '/post_img/190902/style_transfer_5.png'/>
_연세대학교 응용통계학과 대학원생 S군_

이 모델을 Video에도 적용해 보았다. training 시간은 프레임당 약 70초로, 20초짜리 동영상(약 500 Frames)을 Training 시키는데 약 10시간이 걸렸다.

<img src = '/post_img/190902/Lua_1.gif' width="200"/>
<img src = '/post_img/190902/Lua_2.gif' width="200"/>
<img src = '/post_img/190902/Lua_3.gif' width="200"/>

가장 왼쪽의 Video가 원본 영상이며, 차례대로 '벼랑위의포뇨'와 '김홍도' 스타일로 해당 영상을 변환해본 모습이다. 나름 괜찮은 성능을 보여서 깜짝 놀랐다. 하지만 다양한 input videos를 사용해보던 중 문제점을 발견하게 되었다. 다음 output을 보자.



<br>







<img src = '/post_img/190902/main.gif'/>
