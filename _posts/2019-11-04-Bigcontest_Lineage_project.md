---
layout: post
title: 2019 빅콘테스트(BigContest)-리니지 유저 이탈 예측 project
subtitle: Bigcontest
category: Project
use_math: true
---

## 2019 빅콘테스트 챔피언리그

<br>

<center><img src = '/post_img/191104/bigcontest_main.png' width="600"/></center>

___<center>2019 Bigcontest </center>___

대학원 친구들, 학부생 후배들과 함께 2019 Bigcontest에 참가했다.

<br>

<center><img src = '/post_img/191104/image0.png' width="600"/></center>

우리가 참여한 대회는 2019 빅콘테스트의 챔피언리그였다. 챔피언리그의 경우, 엔씨소프트에서 제공하는 ‘리니지’ 고객 활동 데이터를 활용하여 향후 고객 이탈 방지를 위한 프로모션 수행 시 예상되는 잔존가치를 산정하는 예측 모형을 개발하는 것이 대회의 목표였다.

<br>
<br>
## 1차 예선

평소에 게임을 좋아하다보니 이러한 주제가 매우 흥미로웠다. 8월부터 약 2달 간 모델링에 전념하여 자체평가 리더보드에서 상위권에 안착할 수 있었다. 하지만 실제 test dataset의 20%만 사용한 결과이므로, 실제 순위는 알 수 없었다. 1차 심사까지 마음을 졸이면서 기다렸고, 감사하게도 무난히 예선을 통과했다.

우리 팀 모델의 Architecture는 다음과 같다.

<br>

<center><img src = '/post_img/191104/image6.png' width="600"/></center>

XGBoost, LightGBM, RandomForest과 같은 다양한 모델을 기반으로하여 예측 모형을 모델링하였으며, 예측 값에 대한 scaling을 통하여 점수를 극대화하고, 앙상블을 통한 regularization까지 신경써주었다.

사실 지나고보니 무작정 예측 모델을 만들어보는 것보다는, 실제로 게임을 플레이해보면서 Domain Knowledge를 이해하였던 경험이나, 출제자의 의도가 무엇인지 파악하려고 노력했던 점이 예선 통과에 중요하게 작용했던 것 같다.


<br>
<br>
## 2차 본선

판교 엔씨소프트R&D센터에서 2차 본선이 열렸다. 약 15분간 발표시간이 주어졌고, 우리의 모델에 대한 심사위원님들의 다양한 피드백을 들을 수 있어서 좋았다. 2차 본선을 통과하면 최소 장려상 수상이었기 때문에 매우 떨렸고, 긴장되었다.

발표가 끝난 이후에는 팀원 모두 떨어진 줄 알고 우울해 했었는데, 너무나도 감사하게도 2차 본선 통과 소식을 듣게 되었다.

<br>
<br>
## 시상식

<br>

<center><img src = '/post_img/191104/image1.png' width="600"/></center>


2019 데이터 진흥주간에서 11월 26일(화)에 시상식이 열렸다. 2차 본선을 통과하여 수상이 확정되었으나, 어떤 상을 받게 될지는 미공개인 상태로 시상식에 참가하게 된다.

<br>

<center><img src = '/post_img/191104/image2.png' width="600"/></center>

이 날은 시상식 뿐만아니라, 빅데이터 기반 모델링을 활용하고 있는 여러 업체들이 참여하여 다양한 행사가 열렸다. 구경하는 재미도 쏠쏠했다.

<br>

<center><img src = '/post_img/191104/image3.png' width="600"/></center>

이곳 저곳 구경하고 다니다가, 시상식 시간이 되어 시상식장 앞으로 찾아갔다. 드디어 시작된 시상식.

<br>

<center><img src = '/post_img/191104/image5.png' width="600"/></center>

수상을 하게된 것만으로 너무나도 감사했는데, 장려상이 아닌 __우수상(빅데이터포럼의장상)__ 을 수상하였다. 약 두 달간 온갖 정성을 쏟아 만든 우리 팀의 예측모형과 발표자료를 생각했던 것 이상으로 평가받게되어 너무나도 감사하고, 뿌듯했다.

<br>

<center><img src = '/post_img/191104/image4.png' width="600"/></center>

거의 세 달을 통계연구소 연구실에서 모여 밤을 새가며 함께 고생한 우리팀화이팅 멤버들 사진이다. 너무나도 소중한 경험을 할 수 있게 만들어준 팀원들에게 감사한다.

<br>
<br>
## 발표자료

[2019 빅콘테스트 우리팀화이팅 발표자료](https://kjhov195.github.io/post_img/191104/%EC%9A%B0%EB%A6%AC%ED%8C%80%ED%99%94%EC%9D%B4%ED%8C%85.pdf)


<br>
<br>
