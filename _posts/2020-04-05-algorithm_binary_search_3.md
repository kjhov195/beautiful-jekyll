---
layout: post
title: 프로그래머스(Algorithm)-징검다리
subtitle: Algorithm, Binary Search
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200405/image2.png' width="600"/></center>
<center><img src = '/post_img/200405/image3.png' width="600"/></center>


<br>
### 풀이

```
def solution(distance, rocks, n):
    rocks.append(distance)
    rocks = sorted(rocks)
    left = 0
    right = rocks[-1]
    answer = 0

    while left<=right:
        mid = int((left+right)/2)

        prev = 0
        cnt = 0
        betweenList =[]
        for x in rocks:
            between = x-prev
            if between<mid:
                cnt += 1
            else:
                prev = x
                betweenList.append(between)
        betweenMin = min(betweenList)

        if cnt>n:
            right = mid-1
        else:
            if answer<betweenMin:
                answer = betweenMin
            left = mid+1
    return answer
```

<br>
### Reference

https://contest.usaco.org/DEC06.htm
