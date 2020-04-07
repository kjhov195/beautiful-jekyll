---
layout: post
title: 프로그래머스(Algorithm)-입국심사(Binary Search)
subtitle: Algorithm, Binary Search
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200405/image2.png' width="600"/></center>
<center><img src = '/post_img/200405/image3.png' width="600"/></center>


<br>
### 풀이

알고리즘 문제들을 풀면서 가장 많이 반성하게 된 문제이다.

처음에는 n을 한 명씩 줄여가면서 심사관들에게 배정하는 방식으로 문제를 풀려고 했고, 이 생각에 사로잡혀서 거의 한 시간 동안 묶여 있었던 것 같다. 사실 이 문제는 그렇게 풀면 답을 구할 수가 없다.(아마도)

이 문제는 특정 시간을 준 후에, 심사관들이 이 시간 안에 해결할 수 있는 입국 심사자들 수의 합을 구하는 방법으로 풀어야 한다.

그 합이 n보다 클 경우 시간을 조금 더 줄여가고, 그 합이 n보다 작을 경우 시간을 조금 더 늘려가는 방향으로 이분 탐색을 활용하여 문제를 풀어야 한다.

```
def solution(n, times):
    times = sorted(times)
    left = 1
    right = times[-1]*n
    answer = right
    while left<=right:
        mid = int((left+right)/2)
        total = sum([int(mid/x) for x in times])
        if total<n:
            left = mid+1
        else:
            if mid<answer:
                answer = mid
            right = mid-1
    return answer
```

<br>
### Reference

http://hsin.hr/coci/archive/2012_2013/contest3_tasks.pdf
