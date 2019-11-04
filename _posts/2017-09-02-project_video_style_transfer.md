---
layout: post
title: Animation Generation Project
subtitle: Video Style Transfer, Edge Detection
category: Project
use_math: true
---

### 2019 Spring, YBIGTA Conference 1st
2019 YBIGTA Conference에서 우리 팀이 1등을 수상하였다. 우리 팀의 프로젝트 주제는 현실 영상을 애니메이션 스타일로 변환해주는 모델을 만들어보는 것이었다.

<img src = '/post_img/190902/style_transfer_1.png'/>
_Loving Vincent(2017)_

2017년에 개봉하여 흥행에 성공했었던 _Loving Vincent(2017)_ 에서 아이디어를 착안하여 팀원들과 이 프로젝트를 시작하게 되었다. 러빙빈센트의 경우 107명의 화가가 약 2년 동안 65,000프레임에 달하는 양을 작화했다. 실제 영화를 찍은 뒤, 이를 그림으로 그려내는 작업에 아주 많은 시간과 자원, 노력이 들어간 것이다. 만약 Style Transfer Model이 잘 작동하여 이를 수행해준다면, 혹은 그림으로 그려내는 작업의 밑바탕이 되어준다면 좋지 않을까라는 생각이었다.

<br>
<br>
### Model searching

사실 작년에 KT ATC에서 다양한 프로젝트를 진행하면서 Image에 대한 Style Transfer Model(Vanilla gan, DCGAN, Cycle GAN 등)은 많이 접해보았지만, Video의 Style Transfer는 처음 접해보는 영역이었다.

처음 떠오른 모델은 GAN이였고, 역시 최종적으로 사용한 모델 또한 GAN이었다. 하지만 GAN의 치명적인 단점인 _동일한 image를 input으로 주어도 styled된 결과가 매번 달라지는 단점_ 때문에 조금 더 안정적인 모델이 있지 않을까 하는 생각에 다른 model 또한 searching해 보았다.

모델 Searching 과정에서 NVIDIA & MIT의 Video-to-Video Synthesis 등의 모델을 찾았지만 Paired Dataset에 대해서만 학습이 가능하다는 한계가 있었다. 우리 팀이 직접 Paired Dataset을 만들 수는 없는 상황이었으므로 안타깝게도 이 모델은 사용할 수 없었다. GAN based model이 아닌 다른 여러 Style Transfer 모델들 또한 Video가 아닌 Image에 대한 연구가 대부분이었다.

결국 기본 Base가 되는 모델로 GAN을 선택하게 되었다.

<br>
<br>
### Style Transfer

<img src = '/post_img/190902/style_transfer_2.png'/>
_Loving Vincent(2017)_

다양한 Video Style Transfer 모델을 시도해보면서 LUA와 Pytorch로 구현된 여러 모델들로 Video Styel Transfer를 시도해보았다.





<img src = '/post_img/190902/main.gif'/>
