---
layout: post
title: 프로그래머스(Algorithm)-라면 공장(Heap)
subtitle: Algorithm, Heap
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200407/image3.png' width="600"/></center>
<center><img src = '/post_img/200407/image4.png' width="600"/></center>


<br>
### 풀이

오름차순으로 항상 정렬된 상태에서 가장 작은 것을 빼오는 연산을 하고 싶다면 힙(Heap)을 사용하는 것이 좋다.

하지만 만약, 가장 큰 것을 반복적으로 빼오는 연산을 하고 싶다면 어떻게 할까?

Heap에 push할 때 Minus(-)를 붙여서 ```heappush()```한 뒤, ```heappop()```으로 가장 작은 것을 꺼내와서 다시 -를 붙여주는 방법으로 가장 큰 수를 찾아낼 수 있다.

```
import heapq

def solution(stock, dates, supplies, k):
    cnt = 0
    start = 0
    n = len(dates)

    plan = []
    heapq.heapify(plan)
    while stock<k:
        for i in range(start,n):
            if dates[i]<=stock:
                heapq.heappush(plan,-supplies[i])
            else:
                break
        stock += -heapq.heappop(plan)
        start = i
        cnt += 1
    return cnt

```
