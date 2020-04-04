---
layout: post
title: Algorithm-다리를 지나는 트럭
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image3.png' width="600"/></center>
<center><img src = '/post_img/200404/image4.png' width="600"/></center>

<br>
### 풀이

```
def solution(bridge_length, weight, truck_weights):
    n = len(truck_weights)
    q = [0]*bridge_length
    t = 0
    idx = 0

    while True:
        t+=1
        w = truck_weights[idx]
        q.pop(0)
        if sum(q)+w<=weight:
            q.append(w)
            idx += 1
        else:
            q.append(0)

        if idx == n:
            break

    answer = t+bridge_length
    return answer
```

<br>
### Reference

http://icpckorea.org/2016/ONLINE/problem.pdf
