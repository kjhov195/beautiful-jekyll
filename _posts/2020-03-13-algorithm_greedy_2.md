---
layout: post
title: Algorithm-조이스틱
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200313/image3.png' width="600"/></center>
<center><img src = '/post_img/200313/image4.png' width="600 "/></center>

<br>
### 풀이

```
def solution(name):
    d = {chr(64+i): i for i in range(1,27)}

    nameSplit = [x for x in name]

    toMove = []
    for x in nameSplit:
        toMove.append(min(d[x]-1,26-d[x]+1))

    cnt = 0
    i = 0
    while True:
        cnt += toMove[i]
        toMove[i] = 0

        if sum(toMove) == 0:
            break

        l,r = 1,1

        while toMove[i-l] == 0:
            l += 1

        while toMove[i+r] == 0:
            r += 1

        if l<r:
            cnt += l
            i -= l
        else:
            cnt += r
            i += r
    answer = cnt
    return answer
```

<br>
### Reference

https://commissies.ch.tudelft.nl/chipcie/archief/2010/nwerc/nwerc2010.pdf
