---
layout: post
title: 프로그래머스(Algorithm)-카펫(Brute-force)
subtitle: Algorithm, Brute-force
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image11.png' width="600"/></center>
<center><img src = '/post_img/200313/image12.png' width="600"/></center>

<br>
### 풀이

```
def solution(brown, red):
    x, y = 3, 3
    while True:
        calcRed = (x-2)*(y-2)
        calcBrown = x*y-red

        if calcRed == red and calcBrown == brown:
            break
        elif x*y>red+brown:
            y += 1
            x = y
        else:
            x += 1
    answer = [x,y]

    return answer
```

<br>
### Reference

https://hsin.hr/coci/archive/2010_2011/contest4_tasks.pdf
