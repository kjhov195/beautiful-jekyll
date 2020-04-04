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
### 오답

이 풀이는 케이스 하나에서 시간초과가 떴다. 그 이유를 한참 찾았는데, 이유는 ```sum()```함수 때문이었다.

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

초기 s 값을 0으로 두고, 큐에서 pop할 때마다 s에서 빼주고, 큐에 추가할 때마다 s에 더해주는 방식으로 합을 구했다. 앞서 오답처리 되었던 케이스도 이제는 정답처리가 된다.

```
def solution(bridge_length, weight, truck_weights):
    n = len(truck_weights) #트럭 수
    q = [0]*bridge_length  #큐(길이 n)
    t = 0                  #시간
    s = 0                  #다리 위 무게 합
    idx = 0                #트럭 index
    while True:
        t+=1
        w = truck_weights[idx]
        temp = q.pop(0)
        s -= temp
        if s+w<=weight:
            q.append(w)
            s += w
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
