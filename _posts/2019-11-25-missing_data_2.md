---
layout: post
title: Missing data(2)-Statistical model
subtitle: Missing data
category: Statistics
use_math: true
---

<br>
<br>

앞서 Missing data(1) 포스트에서 고혈압 환자의 혈압 측정 예시를 통하여 Missing data에 대한 직관적인 이해를 중심으로 살펴보았다. 이번 포스트에서는 수식을 통하여 조금 더 구체적으로 Missing data와 modeling에 대하여 살펴보도록 하겠다.

<br>
<br>
### Definition

$$
\begin{align*}
Y &= (Y_o, Y_m)\\
f(Y\vert \theta) &= f(Y_0, Y_m \vert \theta)\;:\;\text{joint pdf of $Y_o$, $Y_m$}\\\\
,where \;&\begin{cases}
Y_o &: \text{observed values}\\
Y_m &: \text{missing values}\\
\end{cases}
\end{align*}
$$

y 값은 관측된($Y_o$:observed) y값과 관측되지 않은 y값($Y_m$:missing)으로 나눌 수 있다. 현실에서는 관측되지 않은 $Y_m$의 값은 알 수 없다.(관측되지 않았으므로 당연하다.)

우리가 알 수 있는 것은 $Y_m$이 아닌, 결측되었는지에 대한 여부 뿐이다. 따라서 다음과 같은 M matrix를 정의하여 결측 데이터를 나타낸다.

$$
\begin{align*}
\underset {n \text{ by } k} Y = (y_{ij})
\end{align*}
$$



<br>
<br>
### MAR

writing...

<br>
<br>
### Reference
강승호(2019), 신약개발에 필요한 임상통계학, 자유아카데미


<br>
<br>
