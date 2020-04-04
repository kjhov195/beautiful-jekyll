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
### Python의 강점

Python의 강점인 list comprehension을 활용하여 풀면 다음과 같이 짧고 간결하게 나타낼 수 있다.(짧지만, 이 알고리즘 또한 앞서 풀었던 풀이와 똑같은 방식으로 문제를 해결한다.)

```
def solution(array, commands):
    return [sorted(array[i-1:j])[k-1] for i,j,k in commands]
```

<br>

반면 C++로 풀 경우, 다음과 같이 풀어야 한다.

```
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> solution(vector<int> array, vector<vector<int>> commands) {
    vector<int> answer;
    vector<int> temp;

    for(int i = 0; i < commands.size(); i++) {
        temp = array;
        sort(temp.begin() + commands[i][0] - 1, temp.begin() + commands[i][1]);
        answer.push_back(temp[commands[i][0] + commands[i][2]-2]);
    }

    return answer;
}
```

Python의 특징과 장점을 잘 느낄 수 있는 문제인 것 같다.

<br>
### Reference

https://neerc.ifmo.ru/subregions/northern.html
