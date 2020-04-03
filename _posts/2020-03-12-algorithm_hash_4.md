---
layout: post
title: Algorithm-베스트앨범
subtitle: Algorithm, Hash
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200312/image7.png' width="450"/></center>
<center><img src = '/post_img/200312/image8.png' width="450"/></center>

<br>
### 풀이

```
def solution(genres, plays):
    idxs = [i for i in range(len(genres))]

    total = list(zip(idxs,genres,plays))
    total_sort = sorted(total, key = lambda x:x[2], reverse = True)

    d1 = {}
    for x,y,z in total_sort:
        d1[y] = d1.get(y,0) + 1
    d1_sort = sorted(d1.items(), key = lambda x:x[1], reverse = True)

    d2 = {}
    for x,y,z in total_sort:
        d2[y] = d2.get(y,0) + z
    d2_sort = sorted(d2.items(), key = lambda x:x[1], reverse = True)

    d1_dict = dict(d1_sort)

    answer = []
    for k, v in d2_sort:
        i = 0
        cnt = 0
        if d1_dict[k] == 1:
            while cnt < 1:
                temp = total_sort[i]
                if temp[1] == k:
                    answer.append(temp[0])
                    cnt += 1
                else:
                    pass
                i += 1
        else:
            while cnt < 2:
                temp = total_sort[i]
                if temp[1] == k:
                    answer.append(temp[0])
                    cnt += 1
                else:
                    pass
                i += 1
    return answer
```

<br>
### Reference

https://programmers.co.kr/learn/courses/30/lessons/42579
