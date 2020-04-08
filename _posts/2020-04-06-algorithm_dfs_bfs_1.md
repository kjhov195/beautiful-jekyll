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

DFS를 활용하여 푸는 문제다. 재귀함수로 푸는 방법은 잘 이해가 안돼서.. 스택을 활용하여 풀어보았다.

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
