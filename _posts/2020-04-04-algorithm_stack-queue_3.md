---
layout: post
title: 프로그래머스(Algorithm)-기능개발
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image5.png' width="600"/></center>
<center><img src = '/post_img/200404/image6.png' width="600"/></center>

<br>
### 풀이

```
def solution(progresses, speeds):
    n = len(progresses)
    answer = []
    complete = [0]*n
    bind = list(zip(progresses,speeds))
    bindUpdate = bind

    while complete:
        bindUpdate = [(x+y,y) for x,y in bindUpdate]
        for i in range(len(bindUpdate)):
            if bindUpdate[i][0] >= 100:
                complete[i] = 1
        cnt = 0
        while complete and complete[0] == 1:
            complete.pop(0)
            bindUpdate.pop(0)
            cnt += 1
        if cnt > 0:
            answer.append(cnt)

    return answer
```

<br>
### Reference

http://icpckorea.org/2016/ONLINE/problem.pdf
