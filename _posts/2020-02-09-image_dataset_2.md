---
layout: post
title: Image Dataset(2)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

이번 포스트에서는 다양한 Computer Vision tasks에서 사용되는 데이터셋들에 대하여 살펴보도록 하겠다.


<br>
<br>
### COCO

<br>

<center><img src = '/post_img/200209/image10.jfif' width="450"/></center>

paper: [Tsung-Yi Lin Et al.(2015), Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf)

COCO는 Common Object in Context의 줄임말로서, object detection, keypoint detection, stuff segmentation, panoptic segmentation, image captioning을 위한 데이터셋이다.

Microsoft의 COCO Dataset은 약 33만개의 데이터로 구성되어 있으며, 여러 버전에 걸쳐 공개되었다.

MS COCO paper에는 91개의 class라고 명시되어 있지만, 실제로는 80개의 class가 제공된다.

<br>
<br>
### Pascal VOC

<br>

<center><img src = '/post_img/200209/image11.png' width="450"/></center>

paper: [Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn(2010), Andrew ZissermanThe PASCAL Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

Pascal VOC는 object detection을 위한 데이터셋이다.

2005년부터 2012년까지 Pascal VOC Challenge가 열렸으며, 이 대회에 사용된 데이터셋이다. 현재는 더이상 진행되지 않지만, 여전히 많이 사용되고 있는 데이터셋이다.

대회는 종료되었지만, 여전히 PASCAL VOC Evaluation Server(http://host.robots.ox.ac.uk:8080/)에서 Pascal VOC 데이터셋에 대한 evaluation 결과를 확인할 수 있다.


<br>
<br>
### Reference

[COCO](http://cocodataset.org/)

[The PASCAL Visual Object Classes Homepage](http://host.robots.ox.ac.uk/pascal/VOC/index.html)
