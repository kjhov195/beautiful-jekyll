---
layout: post
title: Algorithm-다리를 지나는 트럭
subtitle: Algorithm, Stack/Queue
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<center><img src = '/post_img/200404/image3.png' width="600"/></center>
<center><img src = '/post_img/200404/image4.png' width="600"/></center>

<br>
### 풀이 1

```
def solution(bridge_length, weight, truck_weights):
    n = len(truck_weights)
    q = [0]*bridge_length
    t = 0
    idx = 0

    while True:
        t+=1
        w = truck_weights[idx]
        q.pop(0)
        if sum(q)+w<=weight:
            q.append(w)
            idx += 1
        else:
            q.append(0)

        if idx == n:
            break

    answer = t+bridge_length
    return answer
```

<br>
### 풀이 2

```
def solution(bridge_length, weight, truck_weights):
    answer = 0
    curr_weight = 0
    # 1. 스택 생성
    st = truck_weights[::-1]
    # 2. 큐 생성
    q = []
    # 3. 진행 거리 배열 생성
    dist = [0] * len(truck_weights)
    # 4. 마지막 트럭이 다리를 지날때까지 다음을 반복합니다.
    while st:
        # 1. 출발해야 할 트럭 top을 구합니다. 즉, st에서 데이터를 빼옵니다.
        top = st.pop()

        # 2. 현재 다리를 지나는 트럭들의 무게와, top의 무게를 더한 값이 weight보다 큰지 확인 합니다.
        # 2-1. 크다면, 현재 트럭은 다리를 지나지 않습니다. 다시 스택으로 되돌립니다.
        # 2-2. 그렇지 않다면, 트럭은 다리를 지납니다. 즉 큐에 데이터를 넣고 다리를 지나는 트럭의 무게를 더합니다.
        if curr_weight + top > weight:
            st.append(top)
        else:
            curr_weight += top
            q.append(top)

        # 3. 현재 다리에 들어선 트럭들을 움직입니다. 즉, 각 진행 거리를 나타내는 dist를 q의 길이만큼 순회하여 1씩 더해줍니다.
        for i in range(len(q)):
            dist[i] += 1

        # 4. 다리를 지난 트럭을 제거합니다. 진행 거리가, 입력 bridge_length보다 큰지 확인합니다.
        # 4-1. curr_weight에서 q의 첫 원소만큼 빼주고, q에서 그 데이터를 제거합니다.
        # 4-2. dist 역시 첫 원소를 제거해주어야 합니다.
        if dist[0] >= bridge_length:
            curr_weight -= q.pop(0)
            dist.pop(0)

        answer += 1

    # 5. 마지막 트럭의 진행한 거리를 구합니다.
    length = dist.pop()
    # 6. answer 에 마지막 트럭이 다리를 지나는 시간 (다리 길이 - 현재 진행한 길이 + 1)을 더합니다.
    answer += (bridge_length - length + 1)
    return answer
```

<br>
### Reference

http://icpckorea.org/2016/ONLINE/problem.pdf

https://gurumee92.tistory.com/168?category=782306
