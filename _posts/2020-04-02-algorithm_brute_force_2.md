---
layout: post
title: 프로그래머스(Algorithm)-소수 찾기(Brute-force)
subtitle: Algorithm, Brute-force
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image8.png' width="600"/></center>

<br>
### 풀이

```
from itertools import permutations
def solution(numbers):
    numList = list(numbers)
    perList = []
    for i in range(1,len(numList)+1):
        perList += list(permutations(numList,i))
    perListUnique = list(set([int(''.join(x)) for x in perList]))
    perListUnique = list(set([int(''.join(x)) for x in perList]))

    cnt = 0
    while perListUnique:
        flag = True
        top = perListUnique.pop()
        if top<=1:
            flag = False
        for i in range(2,top):
            if top%i==0:
                flag = False
        if flag:
            cnt += 1
    answer = cnt
    return answer
```

<br>
### Reference

http://2009.nwerc.eu/results/nwerc09.pdf
