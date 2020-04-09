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

<br>

다른 분께서 푼 방법인데, 다음과 같이 재귀를 활용하여 DFS를 구현하여 풀 수도 있다. 출제자의 의도를 잘 반영하여 푼 것 같다.

```
def solution(numbers, target):
    if not numbers and target==0:
        return 1
    elif not numbers:
        return 0
    else:
        return solution(numbers[1:],target-numbers[0])+solution(numbers[1:],target+numbers[0])
```
