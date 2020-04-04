---
layout: post
title: Algorithm-구명보트
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200314/image1.png' width="600"/></center>

<br>
### 풀이

정확성 테스트에서는 통과하는데, 일부 20%정도의 효율성 테스트에서 통과하지 못하는 경우도 있었다.

그 이유는 아마도 ```pop(0)```연산이 첫 번째 index의 element를 지운 후, 한 칸씩 앞으로 당기는 연산이기 때문에 __O(1)__이 아닌 __O(n)__ 이 되기 때문인 것 같다.

정확성 테스트를 마저 해결하려면 collections.deque()로 만들어 popleft()를 사용하면 될 듯 하다.

```
def solution(people, limit):
    people.sort(reverse=True)

    idx = 0
    cnt = 0
    n = len(people)
    while idx<n-1:
        if people[idx]+people[-1]<=limit:

            people.pop(idx)
            people.pop()
            idx = 0
            cnt += 1
        else:
            idx += 1
        n = len(people)

    answer = cnt + len(people)
    return answer
```

<br>
### Reference

https://programmers.co.kr/learn/courses/30/lessons/42885
