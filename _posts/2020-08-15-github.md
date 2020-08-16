---
layout: post
title: 프로그래머스(Algorithm)-디스크 컨트롤러(Heap)
subtitle: Algorithm, Heap
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200407/image5.png' width="600"/></center>
<center><img src = '/post_img/200407/image6.png' width="600"/></center>
<center><img src = '/post_img/200407/image7.png' width="600"/></center>


<br>
### 풀이 1

```
import heapq
def solution(jobs):
    n = len(jobs)
    cs = [[y,x] for x,y in jobs]
    csSort = sorted(cs,key=lambda x:x[1],reverse=True)
    listWait = []
    heapq.heapify(listWait)
    step = 0
    t=-1
    flag = False
    answer = []
    while len(answer)<n:
        t+=1
        step +=1

        if csSort:
            top = csSort.pop()
            if top[1]==t:
                heapq.heappush(listWait,top)
            else:
                csSort.append(top)

        if flag and takes == step:
            answer.append(now+[t])
            flag=False
        if not flag and listWait:
            now = heapq.heappop(listWait)+[t]
            takes = now[0]
            step = 0
            flag = True
    return int(sum([w-y for x,y,z,w in answer])/n)
```

처음에 짤 때는 시간이 부족해서 일단 생각나는대로 짜서 제출해보았다. 참고로 이 코드를 제출하면 다음과 같은 처참한 결과를 확인할 수 있다.

<center><img src = '/post_img/200407/image8.png' width="600"/></center>


<br>
<br>
### 풀이 2

```
import heapq
from collections import deque

def solution(jobs):
    tasks = deque(sorted([(x[1], x[0]) for x in jobs], key=lambda x: (x[1], x[0])))
    q = []
    heapq.heappush(q, tasks.popleft())
    current_time, total_response_time = 0, 0

    while len(q) > 0:
        dur, arr = heapq.heappop(q)
        current_time = max(current_time + dur, arr + dur)
        total_response_time += current_time - arr

        while len(tasks) > 0 and tasks[0][1] <= current_time:
            heapq.heappush(q, tasks.popleft())

        if len(tasks) > 0 and len(q) == 0:
            heapq.heappush(q, tasks.popleft())
    return total_response_time // len(jobs)
```

<center><img src = '/post_img/200407/image9.png' width="600"/></center>
