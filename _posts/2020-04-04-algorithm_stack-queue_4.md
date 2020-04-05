---
layout: post
title: Algorithm-프린터
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image7.png' width="600"/></center>
<center><img src = '/post_img/200404/image8.png' width="600"/></center>

<br>
### 풀이

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
### Reference

http://www.csc.kth.se/contest/nwerc/2006/problems/nwerc06.pdf
