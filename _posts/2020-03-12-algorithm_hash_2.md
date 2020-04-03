---
layout: post
title: Algorithm-전화번호 목록
subtitle: Algorithm, Hash
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200312/image3.png' width="450"/></center>
<center><img src = '/post_img/200312/image4.png' width="450"/></center>

<br>
### 이해

이문제는 프로그래머스에서 해시 문제 카테고리에 속해 있는 문제이다. 그래서 처음에는 해시를 활용하여 문제를 풀어보려고 했는데, 굳이 해시를 사용하여 푸는 방법이 생각나지 않았다.

그것보다는 list를 활용하는게 훨씬 더 쉽겠다는 생각이 들어 그렇게 풀었다. 아마도 해시로 풀면 효율성 측면에서 훨씬 더 좋을 것 같긴 하다.

<br>
### 풀이

```
def solution(phone_book):
    for x in phone_book:
        temp_pb = phone_book.copy()
        temp_pb.remove(x)
        l = len(x)
        for y in temp_pb:
            if y[:l] == x:
                return False
    return True
```

<br>
### Reference

https://nordic.icpc.io/
