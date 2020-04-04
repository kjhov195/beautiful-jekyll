---
layout: post
title: Algorithm-큰 수 만들기
subtitle: Algorithm, Greedy
category: Data Structure, Algorithm
use_math: true
---

<br>
### 문제

<br>
<center><img src = '/post_img/200313/image5.png' width="600"/></center>

<br>
### 풀이

이 문제는 만들 수 있는 숫자를 모두 만든 뒤, 가장 큰 숫자를 찾는 방법으로 찾을 수도 있다. 하지만 그럴 경우 가능한 조합의 수가 $_{n} C_{n-k}$이며, 이 때문에 효율성 에러가 발생하게 된다. 이 경우, 시간복잡도는 __O(n)__ 이 된다.

(참고로 이렇게 문제의 조합(Combination)이 입력/제약/경계에 의해 영향을 받아 문제의 복잡성이 급격하게 증가하는 것을 조합적 폭발(Combinatorial Explosion)이라고 부른다.)




<br>
### Reference

https://hsin.hr/coci/archive/2011_2012/contest4_tasks.pdf
