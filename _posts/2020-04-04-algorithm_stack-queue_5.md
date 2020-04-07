---
layout: post
title: 프로그래머스(Algorithm)-쇠막대기(Queue)
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image9.png' width="600"/></center>
<center><img src = '/post_img/200404/image10.png' width="600"/></center>

<br>
### 풀이

```
def solution(arrangement):
    cnt = 0
    sticks = 0
    lRazor = arrangement.replace("()",'R')

    for x in lRazor:
        if x == "(":
            sticks += 1
        elif x== ")":
            sticks -= 1
            cnt += 1
        else:
            cnt += sticks
    answer = cnt
    return answer
```

<br>
### Reference

https://www.digitalculture.or.kr/koi/selectOlymPiadDissentList.do
