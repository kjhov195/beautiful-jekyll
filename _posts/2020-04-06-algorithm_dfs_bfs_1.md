---
layout: post
title: 프로그래머스(Algorithm)-타겟 넘버(DFS/BFS)
subtitle: Algorithm, DFS, BFS
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200406/image3.png' width="600"/></center>

<br>
### 풀이

이 문제는 DFS를 활용하여 푸는 문제다. Recursive하게 풀고 싶었는데, 잘 안돼서 다르게 풀었다.

```
def solution(numbers, target):
    stack = [[0]]
    level = 0
    while level<len(numbers):
        stack2 = []
        num = numbers[level]
        while stack:
            top = stack.pop()
            stack2.append(top+[num])
            stack2.append(top+[-num])
        stack = stack2
        level += 1
    answer = sum([1 for x in stack if sum(x)==target])
    return answer
```
