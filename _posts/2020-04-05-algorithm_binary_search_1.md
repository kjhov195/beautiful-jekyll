---
layout: post
title: 프로그래머스(Algorithm)-예산
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
    l, r = 1, max(budgets)
    midMax = 0

    while l<=r:
        mid = (l+r)//2
        total = sum([x if x<=mid else mid for x in budgets])
        if total>M:
            r = mid-1
        else:
            l = mid+1
            if midMax < mid:
                midMax = mid
    answer = midMax
    return answer
```

<br>
### 사소한 것

풀다가 문득 궁금해져서 확인해본 내용이다. 정수를 구할 때 ```int((l+r)/2)```로 구하는 것이 빠를까, ```(l+r)//2```로 몫을 구하는 더 빠를까?

직관적으로는 (l+r)//2 연산이 더 빠를 것 같다는 생각이 들었었는데, 실제로는 int()를 썼을 때 빨라지는 경우가 조금 더 많았다.(간혹 더 느려지는 경우도 있었지만..)

##### //2를 쓴 경우

```
테스트 1 〉	통과 (0.10ms, 10.8MB)
테스트 2 〉	통과 (0.09ms, 10.7MB)
테스트 3 〉	통과 (0.09ms, 10.7MB)
테스트 4 〉	통과 (0.09ms, 10.6MB)
테스트 5 〉	통과 (0.08ms, 10.8MB)
테스트 6 〉	통과 (0.11ms, 10.8MB)
테스트 7 〉	통과 (0.07ms, 10.8MB)
테스트 8 〉	통과 (0.08ms, 10.7MB)
테스트 9 〉	통과 (0.09ms, 10.7MB)
테스트 10 〉	통과 (0.07ms, 10.7MB)
테스트 11 〉	통과 (0.11ms, 10.8MB)
테스트 12 〉	통과 (0.13ms, 10.8MB)
테스트 13 〉	통과 (0.08ms, 10.6MB)
테스트 14 〉	통과 (0.11ms, 10.7MB)
테스트 15 〉	통과 (0.05ms, 10.8MB)

테스트 1 〉	통과 (2.72ms, 11.1MB)
테스트 2 〉	통과 (38.94ms, 49.7MB)
테스트 3 〉	통과 (4.79ms, 12MB)
테스트 4 〉	통과 (2.31ms, 11.1MB)
테스트 5 〉	통과 (3.84ms, 11.5MB)
```

##### int를 쓴 경우
```
테스트 1 〉	통과 (0.10ms, 10.7MB)
테스트 2 〉	통과 (0.08ms, 10.8MB)
테스트 3 〉	통과 (0.10ms, 10.7MB)
테스트 4 〉	통과 (0.08ms, 10.7MB)
테스트 5 〉	통과 (0.08ms, 10.7MB)
테스트 6 〉	통과 (0.14ms, 10.7MB)
테스트 7 〉	통과 (0.07ms, 10.7MB)
테스트 8 〉	통과 (0.08ms, 10.7MB)
테스트 9 〉	통과 (0.09ms, 10.8MB)
테스트 10 〉	통과 (0.07ms, 10.7MB)
테스트 11 〉	통과 (0.12ms, 10.7MB)
테스트 12 〉	통과 (0.13ms, 10.8MB)
테스트 13 〉	통과 (0.09ms, 10.6MB)
테스트 14 〉	통과 (0.11ms, 10.8MB)
테스트 15 〉	통과 (0.05ms, 10.8MB)

테스트 1 〉	통과 (1.63ms, 11MB)
테스트 2 〉	통과 (38.89ms, 49.8MB)
테스트 3 〉	통과 (4.79ms, 12MB)
테스트 4 〉	통과 (2.20ms, 11.2MB)
테스트 5 〉	통과 (4.12ms, 11.4MB)
```

<br>
### Reference

https://www.digitalculture.or.kr/koi/selectOlymPiadDissentList.do
