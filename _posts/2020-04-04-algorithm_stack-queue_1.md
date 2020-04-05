---
layout: post
title: 프로그래머스(Algorithm)-탑
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image1.png' width="600"/></center>
<center><img src = '/post_img/200404/image2.png' width="600"/></center>

<br>
### 풀이 1

문제를 처음 봤을 때 생각나는 대로 푼 방법이다.

```
def solution(heights):
    n = len(heights)
    answer = [0]
    for idx in range(1,n):
        h = heights[idx]
        idx2 = idx-1
        while True:
            h2 = heights[idx2]
            if h2 > h:
                answer.append(idx2+1)
                break
            else:
                if idx2 == 0:
                    answer.append(0)
                    break
                else:
                    idx2 -= 1
    return answer
```

<br>
### 풀이 2

프로그래머스의 스택/큐 분류에 속해있는 문제인 만큼 스택을 활용하여 푸는 것이 출제자의 의도에 맞게 푸는 방법일 것이다.

```
def solution(heights):
    answer = []
    st = []

    while heights:
        top = heights.pop()
        while heights and heights[-1] <= top:
            st.append(heights.pop())
        answer.append(len(heights))
        while st:
            heights.append(st.pop())

    answer = answer[::-1]
    return answer
```

<br>
### Reference

https://www.digitalculture.or.kr/koi/selectOlymPiadDissentList.do

https://gurumee92.tistory.com/166
