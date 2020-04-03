---
layout: post
title: Algorithm-위장
subtitle: Algorithm, Hash
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200312/image5.png' width="450"/></center>
<center><img src = '/post_img/200312/image6.png' width="450"/></center>

<br>
### 이해

이 문제에는 옷의 종류와 옷의 이름이라는 2개의 요소로 이루어진 array들의 array(2차원 array)로 이루어져 있다.

서로 다른 옷의 조합의 수를 구해야 하는데, 우선 각 옷의 종류별 unique한 옷의 이름의 수를 구해야 한다. 해당 종류의 옷을 안입는 선택지도 가능하기 때문에, 각 종류별 unique한 옷 이름의 수에 1을 더한 후, 그 수들을 모두 곱해주면 가능한 전체 선택지의 수를 구할 수 있다.

하지만 직전에 입은 스타일과는 다르게 입어야 하므로, 최종적으로 그 수에 1을 빼주어 정답을 구할 수 있다.

이 문제는 dictionary의 get 메써드와 list comprehension을 적절히 사용할 줄 알면 간단하게 풀 수 있는 쉬운 문제다. 파이썬을 처음 접하는 분들이 꼭 직접 풀어보시면 좋을 것 같다.

<br>
### 풀이

```
def solution(clothes):
    d = {}
    for [x,y] in clothes:
        d[y] = d.get(y,0)+1
    vlist = [v+1 for k,v in d.items()]
    mul = 1
    for v in vlist:
        mul *= v
    answer = mul-1
    return answer
```

<br>
### Reference

https://nordic.icpc.io/
