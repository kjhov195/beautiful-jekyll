---
layout: post
title: Algorithm-K번째수
subtitle: Algorithm, Sort
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200314/image4.png' width="600"/></center>
<center><img src = '/post_img/200314/image5.png' width="600"/></center>

<br>
### 풀이

```
def solution(array, commands):
    answer = []
    for l in commands:
        i, j, k = l[0],l[1],l[2]       
        answer.append(sorted(array[i-1:j])[k-1])   
    return answer
```

<br>
### Reference

https://neerc.ifmo.ru/subregions/northern.html
