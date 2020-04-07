---
layout: post
title: 프로그래머스(Algorithm)-모의고사(Brute-force)
subtitle: Algorithm, Brute-force
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image6.png' width="600"/></center>
<center><img src = '/post_img/200313/image7.png' width="600"/></center>

<br>
### 풀이

```
def solution(answers):
    ruleFirst = [1,2,3,4,5]
    ruleSecond = [2,1,2,3,2,4,2,5]
    ruleThird = [3,3,1,1,2,2,4,4,5,5]

    l1 = len(ruleFirst)
    l2 = len(ruleSecond)
    l3 = len(ruleThird)

    scoreFirst = 0
    scoreSecond = 0
    scoreThird = 0
    for i,x in enumerate(answers):
        #first
        if x == ruleFirst[i%l1]:
            scoreFirst += 1
        #second
        if x == ruleSecond[i%l2]:
            scoreSecond += 1
        #third
        if x == ruleThird[i%l3]:
            scoreThird += 1

    scores = [scoreFirst, scoreSecond, scoreThird]
    scores
    maximum = max(scores)

    answer = []
    for i, x in enumerate(scores):
        if x == maximum:
            answer.append(i+1)
    answer.sort()

    return answer
```
