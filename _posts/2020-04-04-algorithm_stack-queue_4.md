---
layout: post
title: 프로그래머스(Algorithm)-프린터(Queue)
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image7.png' width="600"/></center>
<center><img src = '/post_img/200404/image8.png' width="600"/></center>

<br>
<br>
### 풀이 1

특정 문서가 출력되는 순서만 알고싶을 뿐, 전체적인 출력 순서는 알고 싶지 않으므로, ```q```는 구하지 않아도 된다.

```
def solution(priorities, location):
    n = len(priorities)
    # q = [0]*n
    l = []
    idx = [i for i in range(n)]
    while priorities:
        item = priorities.pop(0)
        i = idx.pop(0)
        if priorities and item < max(priorities):
            priorities.append(item)
            idx.append(i)
        else:
            if i == location:
                answer = len(l)+1
                break
            else:
                # q.append(item)
                # q.pop(0)
                l.append(i)
    return answer
```

<br>
<br>
### 풀이 2

풀이 1에서 list, list.append, list.pop(0)를 통해 큐를 만들어 사용할 경우, pop(0)의 연산이 매우 비효율적이다. 이 때, 데크를 활용한다면 효율성 문제를 해결할 수 있다.

데크(Deque: Double-ended queue)는 양방향(앞과 뒤)에서 데이터를 처리할 수 있는 queue형 자료구조이다. 다음 그림은 Deque의 구조를 나타낸다.

<center><img src = '/post_img/200404/Deque.png' width="600"/></center>

<br>

Python의 collections.deque는 list와 비슷하게 사용할 수 있다.(list의 ```append()```, ```pop()```등의 메써드 또한 deque에서 제공한다.)

```
from collections import deque
def solution(priorities, location):
    n = len(priorities)
    l = []
    idx = [i for i in range(n)]

    priorities = deque(priorities)
    idx = deque(idx)

    while priorities:
        item = priorities.popleft()
        i = idx.popleft()
        if priorities and item < max(priorities):
            priorities.append(item)
            idx.append(i)
        else:
            if i == location:
                answer = len(l)+1
                break
            else:
                l.append(i)
    return answer
```

<br>
<br>
### 풀이 1과 풀이 2의 효율성 비교

<br>
#### 풀이 1의 결과

<center><img src = '/post_img/200404/image13.png' width="750"/></center>

<br>
#### 풀이 2의 결과

<center><img src = '/post_img/200404/image14.png' width="750"/></center>

그저 ```list.pop(0)```에서 ```deque.leftpop()```으로만 바꿨을 뿐인데 효율성이 개선되었다.

<br>
<br>
### 풀이 3

다음과 같이 ```deque.rotate()```를 활용할 수도 있다.

<br>
```
from collections import deque
def solution(priorities, location):
    n = len(priorities)
    l = []
    dq = deque(priorities)
    dqIdx = deque([i for i in range(n)])
    cnt = 0

    while dq:
        if dq[0] < max(dq):
            dq.rotate(-1)
            dqIdx.rotate(-1)
        else:
            if dqIdx:
                dq.popleft()
                i = dqIdx.popleft()
                cnt += 1
            else:
                answer = n-1

            if i == location:
                answer = cnt
                break
    return answer
```

<br>
### Reference

http://www.csc.kth.se/contest/nwerc/2006/problems/nwerc06.pdf

https://excelsior-cjh.tistory.com/96
