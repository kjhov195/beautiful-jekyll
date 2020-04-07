---
layout: post
title: 프로그래머스(Algorithm)-여행경로(DFS/BFS)
subtitle: Algorithm, DFS, BFS
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200406/image1.png' width="600"/></center>
<center><img src = '/post_img/200406/image2.png' width="600"/></center>


<br>
### 풀이

한 붓 그리기 문제이며, 스택으로 DFS를 구현하여 풀 수 있다.(한 붓 그리기가 가능하다는 것은 문제에서 보장된다.)

이 문제의 특징은 모든 노드를 방문해야 하는 것이 아니라, 모든 간선을 거쳐야 한다는 것이다.

```
def solution(tickets):
    routes = {}
    for t in tickets:
        routes[t[0]] = routes.get(t[0],[]) + [t[1]]
    for r in routes:
        routes[r].sort(reverse=True)

    stack = []
    stack.append("ICN")
    path = []

    while stack:
        top = stack[-1]
        if top not in routes or len(routes[top])==0:
            path.append(stack.pop())
        else:
            stack.append(routes[top].pop())
    answer = path[::-1]
    return answer
```
