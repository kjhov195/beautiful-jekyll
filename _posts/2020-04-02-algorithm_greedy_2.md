---
layout: post
title: Algorithm-큰 수 만들기
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200313/image5.png' width="600"/></center>

<br>
### 이해

이 문제는 만들 수 있는 숫자를 모두 만든 뒤, 가장 큰 숫자를 찾는 방법으로 찾을 수도 있다. 하지만 그럴 경우 가능한 조합의 수가 $n\;Combination\;(n-k)$이며, 이 때문에 효율성 에러가 발생하게 된다.

(참고로 이렇게 문제의 조합(Combination)이 입력/제약/경계에 의해 영향을 받아 문제의 복잡성이 급격하게 증가하는 것을 조합적 폭발(Combinatorial Explosion)이라고 부른다.)

우리는 이 문제를 다음과 같이 풀고자 한다.

빈 array에 number를 왼쪽부터 하나씩 숫자를 넣되, 이전에 넣은 숫자보다 작거나 같을 경우에만 그냥 넣어준다.

만약 새로 넣을 숫자가 이전에 넣은 숫자보다 더 큰 경우, 이전에 넣은 해당 숫자를 제거하고 새로운 숫자를 넣어준다.(물론 제거할 수 있는 횟수가 남아있는 경우에만.) 이 알고리즘은 시간복잡도 __O(n)__ 을 가진다.

이렇게 __greedy approach를 사용할 수 있는 이유__ 는 __앞 단계에서의 선택이 이후 단계에서의 동작에 의한 해의 최적성(optimality)에 영향을 주지 않기 때문__ 이다.

이러한 경우 가능한 모든 경우의 수를 따지지 않더라도, 한 방향으로 진행하면서 반복적으로 최선의 행동을 선택함으로써 최적의 해를 찾아갈 수 있다.

<br>
### 풀이

```
def solution(number, k):
    l = [int(x) for x in number]

    cnt = 0
    listK = [l[0]]

    idx = 1
    while cnt < k:
        while l[idx]>listK[-1]:
                listK.pop()
                cnt += 1
                if cnt == k:
                    break
                if not listK:
                    break
        listK.append(l[idx])

        if idx == len(number)-1:
            break
        else:
            idx+=1

    if idx<len(number)-1:
        for x in l[idx:]:
            listK.append(x)

    if cnt<k:
        listK = listK[:-(k-cnt)]

    L = [str(x) for x in listK]
    answer = ''.join(L)
    return answer
```

<br>
### Reference

https://hsin.hr/coci/archive/2011_2012/contest4_tasks.pdf
