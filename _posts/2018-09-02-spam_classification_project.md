---
layout: post
title: Spam Classification Project
subtitle: WHOWHO&COMPNAY
category: Project
use_math: true
---

## KT

KT에서의 마지막 프로젝트로 WHOWHO&COMPANY에 파견되어 두 달 간의 실전형 프로젝트를 진행했다.

<br>
<br>
## whowho&compnay

<center><img src = '/post_img/180902/picture_0.png' width="300"/></center>

___<center>WHOWHO&COMPNAY</center>___

KT에서 마지막 프로젝트를 부여받고 발산역에 위치한 후후앤컴퍼니 본사로 출근을 시작했다.

거의 4~5개월을 판교에 위치한 KT로 출퇴근하다가 가까운 발산역으로 출퇴근하기 시작하니 너무나도 좋았다.

<br>

<center><img src = '/post_img/180902/image2.png' width="300"/></center>

후후앤컴퍼니의 주력 사업은 후후앱이다. 후후앱은 발신 전화번호의 정보를 즉시 확인할 수 있는 서비스를 제공한다. 생활에 유용한 전화번호와 악성 전화번호까지 모르는 번호로부터 수신되는 전화번호의 모든 정보를 실시간으로 알려준다.

<br>

<center><img src = '/post_img/180902/image3.png' width="300"/></center>

후후 앱의 가장 큰 강점은 스팸, 보이스피싱 등 악성전화번호를 사전에 식별할 수 있고, 다양한 수신차단 옵션을 통해 불필요한 전화를 완벽히 차단할 수 있다는 것이다. 후후 앱은 안드로이드와 IOS에서 모두 지원된다.

<br>
<br>
## 실시간 스팸 발신자 식별 예측 정확도 개선 프로젝트

후후 프로젝트의 목표는 기존의 실시간 스팸문자 분류 알고리즘을 Deep Learning을 활용하여 개선하는 것이었다. 실제 매출과 연관되는 프로젝트는 처음 맡아봤기에 상당히 흥미로운 프로젝트였다. 프로젝트에서 내가 맡은 역할은 데이터 전처리 및 모델링이었다.

기존에는 heuristic한 규칙을 통한 rule-based 모델을 사용하였다. 이번 프로젝트는 경험적인 정보를 기반으로 한 예측이 아닌, 데이터 분석을 통해 새로운 insight를 얻고, 이를 기반으로 deep learning based 모델 또한 같이 활용하고자 시작되었다.

DNN 기반으로 모델링 하였으며, 기존의 모델과 앙상블한 모델을 사용하여 스팸 분류의 정확도를 높이고자 하였다. 우리가 만든 모형은 데이터분석과 각종 알고리즘을 통해 선별된 37가지 변수를 input으로 사용하며, 스팸문자일 확률을 예측하여 최종적으로 스팸문자 여부를 분류하는 binary classification 문제로 접근하였다. Tensorflow 1.0을 사용하여 모델링하였으며, architecture에 관한 정보는 다음과 같다.

Input shape: [None, 37]
Output shape: [None, 1]
Architecture: FClayer-BN-sigmoid-FClayer-BN-sigmoid
Dropout: p=0.7
Optimizer: Adam with decay rate = 0.9
Initializer: Xavier
Batch size: 100,000


<br>
## 데이터 정제 및 전처리

프로젝트를 진행함에 있어 가장 힘들었던 요소는 전혀 정돈되지 않은 데이터였다. 실제 현업에서 만나게 되는 데이터는 아름답게 정리되어있지 않다는 것을 뼈저리게 느꼈다. 우리가 생각하는 만큼 DB명세서는 완벽하지 않으며, 필요한 데이터는 부족하고 필요하지 않은 데이터가 훨씬 더 많았다.

게다가 통신 쪽의 도메인 지식이 부족하다 보니, 데이터를 이해하는데에도 거의 일주일의 시간이 걸렸던 것 같다. 현업에서 데이터를 다룬다는게 생각보다 쉽지 않은 일이고, 더 많은 데이터를 다루어보고 다양한 분야에 대한 경험을 차근차근 쌓아가야겠다는 생각을 하게 되었다.


<br>
## 팀원

후후앤컴퍼니의 선배님들께서 많이 도와주시고 신경써주셔서 너무나도 감사했고, 행복했던 프로젝트 기간이었다.

마지막은 후후앤컴퍼니에서 팀원, 선배님들과 함께 찍은 마지막 날 사진!! :)

<center><img src = '/post_img/180902/picture_1.png' width="300"/></center>

___<center>WHOWHO&COMPANY, August 1, 2018</center>___

<br>
<br>
<br>
