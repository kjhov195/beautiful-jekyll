---
layout: post
title: 환형 큐(Circular Queue)
subtitle: Data Structure, Algorithm
category: Data Structure, Algorithm
use_math: true
---

<br>
### 환형 큐(Circular Queue)

__정해진 개수__ 의 저장 공간을 빙빙 돌려가며 이용하는 자료구조이다.

환형 큐의 경우 데이터를 집어 넣는 쪽에서는 rear라는 포인터를 가지게 하고, 데이터를 꺼내는 쪽에서는 front라는 포인터를 가지도록 한다. 이렇게 front와 rear를 적절히 계산하여 배열을 환형으로 다시 활용하는 것이 환형 큐의 특징적인 모습이다.

만약 큐가 가득 차면 더이상 원소를 넣을 수 없으므로, 해당 사항을 판단하기 위하여 큐의 길이를 기억하고 있어야 한다. 따라서 Circular Queue의 경우 일반적인 Queue에 비하여 ```isFull()```이라는 연산을 추가적으로 가진다.
