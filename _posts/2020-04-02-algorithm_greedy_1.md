---
layout: post
title: 프로그래머스(Algorithm)-체육복(Greedy)
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image1.png' width="600"/></center>
<center><img src = '/post_img/200313/image2.png' width="600"/></center>

<br>
### 이해

알고리즘 문제를 풀때 언제나 그렇듯, 사람은 어떠한 방식으로 이러한 문제를 푸는가를 생각해보아야 한다. 사람은 통찰력이 있기 때문에, input의 크기가 크지 않다면 한눈에 보고도 이 문제를 풀 수 있다. 그러나 프로그램은 통찰이 아닌, 단계별로 하나하나 차근차근 문제를 풀어갈 수 있도록 알고리즘을 구성해주어야만 이 문제를 해결할 수 있다.

이 문제의 경우, 탐욕법(Greedy Algorithm)을 사용하여 해결할 수 있다.

<br>
### 탐욕법(Greedy Algorithm)

Greedy Algorithm은 __알고리즘의 각 단계에서 그 순간에 최적이라고 생각되는 것을 선택__ 하는 방법을 의미한다.

Greedy Algorithm으로 최적의 해를 찾을 수 있는 경우는 __현재의 선택이 마지막 해답의 최적성을 해치지 않을 때__ 이다.

<br>
### 풀이

이 문제의 경우, 빌려줄 학생들을 __정해진 순서__ 로 살펴야 하고, 이 __정해진 순서__ 에 따라 우선하여 __빌려줄 방향__ 을 정해야 한다.

문제를 풀기위해 빌려줄 수 있는 사람들(```canGive```)와 필요한 사람들(```need```)를 따로 나누었다. 그 후, ```canGive```의 각 원소들을 기준으로 바로 앞을 먼저 확인하여 필요한 사람인지 확인하고, 필요한 사람이라면 그 필요한 사람을 ```need```에서 삭제한다.(빌려주었다고 생각하면, ```need```에 있을 필요가 없으므로.) 만약 바로 앞 사람이 필요한 사람이 아니라면, 바로 뒷 사람을 확인하여 필요한 사람인지 확인하고, 필요한 사람이라면 그 필요한 사람을 ```need```에서 삭제한다.

처음 이 문제를 풀 때 한 가지 실수를 했었는데, ```need```에서만 삭제하면 된다는 것이다. 처음에는 빌려주었으니까 빌려줄 수 있는 사람들(```canGive```)에서도 remove해주었었고, 당연히 이러한 알고리즘은 오답이다. for문이 ```canGive``` 안에서 돌고 있으므로, for문 안에서 ```canGive```를 수정할 경우, for문이 돌고 있는 범위 자체가 달라지기 때문에 건드리면 안된다. 이 점을 주의하자.

```
def solution(n, lost, reserve):
    l = set(lost)
    r = set(reserve)

    canGive = r - l
    need = l - r

    for x in canGive:
        if x-1 in need:
            need.remove(x-1)
        elif x+1 in need:
            need.remove(x+1)
        else:
            pass

    answer = n-len(need)
    return answer
```

<br>
### Reference

https://hsin.hr/coci/archive/2009_2010/contest6_tasks.pdf
