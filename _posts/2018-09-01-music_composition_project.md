---
layout: post
title: Music Composition Project
subtitle: MUSE GAN
category: Project
use_math: true
---

## KT
KT에서 Case Study 마지막 주차에 약 3주간 '노래만드는 AI'라는 프로젝트를 진행했다. KT의 AI 스피커인 기기지니아 탑재될 수 있는 게임, 혹은 다양한 프로그램을 만들어보는 것이 프로젝트의 목표였다.

## 음악 작곡
KT의 기가지니의 경우, 다른 AI 스피커들과는 다르게 집 안의 TV나 기타 다른 기기들과의 연동성이 높은 편이다. 연결된 TV 등의 모니터를 통하여 많은 게임과 다양한 앱을 사용할 수 있다는 것이 기가지니 만의 차별점이다.

KT에서 교육생들에게 기가지니에 탑재할 수 있는 프로그램을 주제로 프로젝트가 주어졌고, 우리 팀은 AI를 활용한 작곡 프로그램을 만들어보기로 했다. Input으로 기존의 음악들을 주면 이를 학습하여 기존에 존재하지 않았던, 하지만 기존의 음악의 스타일을 가진 새로운 음악을 생성해내는 것이 우리 프로젝트의 목표였다.

<img src = '/post_img/180901/musegan_1.png'/>

___프로젝트 흐름___

<br>
<br>
## 모델
'음악 생성', 'GAN'이라는 두 가지 Keyword를 가지고 논문과 모델을 찾기 시작했다. MidiNet과 MuseGAN이라는 두 모델을 찾았고, 해당 모델들을 구현해보았다.

결과적으로 여러 악기들의 MIDI파일을 각각 따로 학습하여 하나의 음악으로 조합해주는 MuseGAN의 음악이 더 듣기 좋고, 풍성한 음악을 생성해냈기 때문에 MuseGAN을 최종 모델로 선택하여 앱을 만들기 시작했다.

논문을 읽고 직접 만들어본 MuseGAN의 Model architecture는 다음과 같다.

<img src = '/post_img/180901/musegan_0.png'/>

___MuseGAN model Architecture___

<br>
#### MIDI(Musical Instrument Digital Interface)
MuseGAN에서 input data는 행렬이다. 논문에 따르면 midi file을 '적절한 전처리'를 거친 뒤 input으로 사용했다고 한다.

MIDI Manufacturers Association(MMA)와 일본의 사단법인 음악전자사업협회(AMEI)가 제정하고 공표한, 전자악기의 연주 데이터를 전송하고 공유하기 위한 업계 표준 규격이다.

언제나 그렇듯, 데이터가 문제였다. 주어진 시간은 3주였으며 예산 또한 주어진 바가 없었다. 공개된 유명한 midi dataset이 있었지만, 우리 프로젝트에는 사용할 수 없는 dataset이었다.

결국 무료로 제공되는 sample midi files를 크롤링하여 dataset을 만들었고 이를 training을 했다. 만약 예산이나 시간이 조금 더 주어졌다면 훨씬 더 좋은 output을 만들어낼 수 있었을 것이라는 아쉬운 생각이 많이 들었다.

고생 끝에 midi file training set을 만들었으나, 전처리가 더 문제였다. 논문을 뒤져봐도, 구글링을 해보아도 midi file을 행렬로 만들 수 있는 방법은 도저히 찾을 수가 없었다. 결국 MuseGAN 저자에게 메일까지 주고받으며 전처리 방법을 알아냈다.

<img src = '/post_img/180901/musegan_2.png'/>

___팀원이 논문 저자와 주고 받은 메일 중 일부___

<br>
#### Piano roll
처음에는 midi file을 바로 numpy array로 변환하는 것이라고 생각했는데, 아무리 찾아봐도 directly 변환할 수 있는 방법이 없었다. 막막한 와중에 논문 저자와 연락을 주고 받으며 piano roll의 형태를 거쳐서 array로 바꿔야 한다는 사실을 알게 되었다.

<img src = '/post_img/180901/musegan_3.png'/>

___piano roll___

piano roll이란 일반적으로는 세로축에 음의 높이, 가로축에 시간으로 연주 정보를 도식화한 것이다. 음의 움직임이나 음표의 길이를 시각적으로 확인하기 쉬운 특징이 있다.

midi 파일을 piano roll로 바꾸기 위해서는 python의 'pretty_midi' 모듈을 사용하면 된다. 전처리에 사용되는 주요 함수는 아래 두 함수이다.

```
pretty_midi.PrettyMIDI() #로컬에 저장된 MIDI file을 load.
get_piano_roll() #load한 MIDI file을 piano roll로 바꾸어 np.array로 저장
```

<br>
#### numpy array
pretty midi를 통하여 numpy array를 얻었다면 numpy array의 shape을 다음과 같이 변환해주어야 한다.

```
(a,b,c,96,84,8)

# a*b*c: number of bars
# 96: beat (time step)
# 84: pitch
# 8: 8 tracks of MUSE GAN(Drums, Piano, Guitar, Bass, Ensemble, Reed, Synth Lead and Synth Pad)
```

piano roll을 적절한 전처리를 통하여 reshape해주는 과정에서 아주 많은 시행착오가 필요했다. 혹시나 MuseGAN에 관심을 가지고 계신 분이라면 reshape해주는 전처리 파일을 github에 업로드 해놓으므로 참고하면 좋을 것 같다.


<br>
<br>
## Output
Input data, Output data 모두 음악 파일이다보니 저작권 문제로 인하여 업로드할 수는 없었다.

원래 모델에서는 8가지 track의 악기들을 모두 사용하여 풍성한 음악을 만들어냈었지만, 우리의 프로젝트에서는 여건 상 한 track의 midi 파일(Piano)을 사용할 수 밖에 없었다. 그래서 그런건지 빠른 템포의 Rock, Ballad 등의 현대적인 음악에 비하여 느리고 조용한 음악이나, Classic 스타일의 output이 굉장히 결과가 좋았다.

<br>
<br>
## setting
MuseGAN을 위한 docker image도 업로드 해놓았으니 필요하신 분이 계시다면 사용하시기 바란다.<br>
<https://hub.docker.com/r/hun1993/mini_musegan/>
