---
layout: post
title: 프로그래머스(Algorithm)-숫자 야구(Brute-force)
subtitle: Algorithm, Brute-force
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image9.png' width="600"/></center>
<center><img src = '/post_img/200313/image10.png' width="600"/></center>

<br>
### 풀이

```
from itertools import permutations
def solution(baseball):
    listPossible = list(permutations([x for x in range(1,10)],3))
    answer = 0
    for cand in listPossible:
        flag = True
        for guessNum,strike,ball in baseball:
            guessNum = tuple(int(x) for x in str(guessNum))
            cntS = 0
            cntB = 0
            for i,x in enumerate(cand):
                if x == guessNum[i]:
                    cntS += 1
                else:
                    if x in guessNum:
                        cntB += 1
            if cntS==strike and cntB==ball:
                pass
            else:
                flag = False
        if flag:
            answer += 1

    return answer
```

<br>
### Reference

https://www.digitalculture.or.kr/koi/selectOlymPiadDissentList.do
