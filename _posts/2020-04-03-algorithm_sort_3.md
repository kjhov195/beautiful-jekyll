---
layout: post
title: 프로그래머스(Algorithm)-H-Index(Sort)
subtitle: Algorithm, Sort
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200314/image6.png' width="600"/></center>

<br>
### 풀이

```
def solution(citations):
    h = len(citations)

    while True:
        if len([x for x in citations if x>=h])>=h:
            break
        else:
            h -= 1
    answer = h
    return answer
```

<br>
### Reference

https://en.wikipedia.org/wiki/H-index
