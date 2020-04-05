---
layout: post
title: Algorithm-주식가격
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image15.png' width="600"/></center>

<br>
### 풀이

list에서 가장 앞의 element를 반복적으로 지워나가야 하는 작업을 할 때는, ```list.pop(0)```를 사용한다거나, list slicing을 활용(예를들어 ```list[1:]```)하지 말고, ```deque.popleft()```를 적극적으로 활용하자.

```
from collections import deque
def solution(prices):
    st = []
    answer = []
    dq = deque(prices)
    while dq:
        bottom = dq.popleft()
        cnt = 0
        for x in dq:
            cnt += 1
            if x<bottom:
                break
        answer.append(cnt)
    return answer
```
