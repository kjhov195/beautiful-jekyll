---
layout: post
title: 프로그래머스(Algorithm)-예산(Binary Search)
subtitle: Algorithm, Binary Search
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200405/image1.png' width="600"/></center>


<br>
### 풀이

어려운 문제는 아니지만, 이분 탐색을 활용하는 것이 익숙하지 않아 풀 때 애를 먹은 문제다.

조금 더 익숙해질 필요가 있을 것 같다.

```
def solution(budgets, M):
    budgetsSort = sorted(budgets)
    left = 1
    right = budgetsSort[-1]
    answer = 0

    while left<=right:
        mid = int((left+right)/2)
        total = sum([x if x<mid else mid for x in budgets])

        if total>M:
            right = mid-1
        else:
            answer = mid
            left = mid+1
    return answer
```

<br>
### 사소한 것

풀다가 문득 궁금해져서 확인해본 내용이다. 정수를 구할 때 ```int((left+right)/2)```로 구하는 것이 빠를까, ```(left+right)//2```로 몫을 구하는 더 빠를까?

직관적으로는 ```(left+right)//2``` 연산이 더 빠를 것 같다는 생각이 들었었는데, 실제로는 ```int((left+right)/2)```를 썼을 때 빨라지는 경우가 조금 더 많았다.(간혹 더 느려지는 경우도 있었지만..)

<br>
### Reference

https://www.digitalculture.or.kr/koi/selectOlymPiadDissentList.do
