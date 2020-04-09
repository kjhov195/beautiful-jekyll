---
layout: post
title: 프로그래머스(Algorithm)-더 맵게(Heap)
subtitle: Algorithm, Heap
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200407/image1.png' width="600"/></center>
<center><img src = '/post_img/200407/image2.png' width="600"/></center>


<br>
### 풀이

```
import heapq

def solution(scoville, K):
    heapq.heapify(scoville)
    cnt = 0
    while len(scoville)>=2:
        cnt += 1
        top1 = heapq.heappop(scoville)
        top2 = heapq.heappop(scoville)
        target = top1+top2*2
        heapq.heappush(scoville, target)
        if scoville[0] >= K:
            return cnt
    return -1
```
