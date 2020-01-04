---
layout: post
title: Multiclass Logistic Regression(Softmax Classifier)-(1)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>
<br>
### Multiclass Logistic Regression, Softmax Classifier

앞서 [Logistic Regression](https://kjhov195.github.io/2020-01-04-logistic_classification/) 포스트에서는 $y$의 class가 두 개인 경우의 logistic regression을 통한 classification에 대하여 살펴보았다.

이번 포스트에서는 $y$의 class가 2개 이상일 때 어떻게 classification을 할 것인지에 대하여 살펴보도록 하겠다.

만약 다루고자 하는 class의 수가 2개 이상일 경우, 우리는 Multiclass Logistic Regression을 다루게 된다. 통계학에서는 Multinomial Logistic Regression, Multiclass Logistic Regression, Multilevl Logistic Regression 등으로 불리우며, 컴퓨터공학이나 머신러닝에서는 Softmax Classifier라고 불리는 경우도 있다. 모두 다 달라보이는 이름이지만, 사실은 똑같은 모형을 일컫는 말이다.

<br>
<br>
### Logistic Regression

Logistic Regression을 통해 Binary classification을 하는 과정에 대해 간단하게 다시 살펴보도록 하자.

우리는 주어지는 input $X$와 weight $W$의 선형 결합 함수 $H_L(X)$를 구할 수 있다.

$$
\begin{align*}
H_L(X) &= XW\\
&=
\begin{bmatrix}
1 & x_1 & x_2 & \cdots & x_p
\end{bmatrix}
_ {1 \times p}
\begin{bmatrix}
b\\
w_1\\
w_2\\
\vdots\\
w_p
\end{bmatrix}_ {p \times 1}\\
&= b + w_1x_1+w_2x_2+\cdots+w_px_p
\end{align*}
$$

다시 처음으로 돌아가서, 우리가 무엇을 하려고 하는지에 대해 생각해보자. 우리가 구하고자 하는 것은 $Y$가 class 1에 속할 확률이다. 하지만 $H_L(\cdot)$ 함수의 치역은 $(-\infty, \infty)$이며, 확률값으로 사용할 수 없는 값을 가진다.

$$ H_L(X) = XW \in (-\infty, \infty)$$

여기서 우리는 $Sigmoid$함수($logistic$함수 라고도 한다)를 사용하여 $H_L(\cdot)$의 값을 0과 1사이의 값으로 변환시킬 수 있게 된다.

$$ \sigma(H_L(X)) = \frac 1 {1+e^{-XW}} \in (0, 1)$$

우리는 Training 과정에서 optimization을 통하여 주어진 데이터를 기반으로 가장 분류를 잘해주는 $W= \hat W$를 찾는다. 이후에 새로운 데이터 $X_{new}$의 class를 predict하고 싶을 때에 다음을 계산하여, 이 값이 cut-off 이상이라면 1로, 이하라면 0으로 예측할 수 있게 된다.

$$\sigma(H_L(X_{new})) = \frac 1 {1+e^{-X_{new} \hat W}}$$


<br>
<br>
### Multiclass Logistic Regression

Multiclass Logistic Regression은 Binary Logistic Regression과 크게 다르지 않으며, 이의 확장이라고 볼 수 있다.

<br>
<br>
### Hypothesis

앞서 2개의 class를 구분하는 Logistic regression의 경우, 하나의 열로 구성된 $W = [w_1 w_2 \cdots w_p]'_ {p \times 1}$ Matrix에 관심을 가지고, 추정해 주었다.

Multiclass Logistic regression의 경우, class 수 만큼의 열을 가진 $W$ matrix를 추정하게 된다. 간단한 설명을 위하여 3개의 class를 가정해보자. 이 때 $W$ matrix는 다음과 같이 $W_{multi}$의 형태로 정의된다.

$$
\begin{align*}
H_L(X) &= XW_{multi}\\
&= X \begin{bmatrix}
W_A & W_B & W_C
\end{bmatrix}\\
&=
\begin{bmatrix}
1 & x_1 & x_2 & \cdots & x_p
\end{bmatrix}
_ {1 \times p}
\begin{bmatrix}
b_A & b_B & b_C\\
w_{A1} & w_{B1} & w_{C1} & \\
w_{A2} & w_{B2} & w_{C2} & \\
\vdots & \vdots & \vdots & \\
w_{Ap} & w_{Bp} & w_{Cp} &
\end{bmatrix}_ {p \times 3}\\
&= \begin{bmatrix}
XW_A & XW_B & XW_C
\end{bmatrix}_ {1 \times 3}
\end{align*}
$$

그 다음 단계에서는 이렇게 구한 linear combination의 값 스칼라 $XW_A$, $XW_B$, $XW_C$에 transformation을 주어 확률 값으로 변환한다.

Binary classification의 경우 sigmoid함수를 사용하여 $\sigma(XW) = \frac 1 {1+e^{-XW}}$와 같이 확률 값으로 변환해 주었다.

Multiclass logistic regression의 경우 sigmoid 함수(logistic 함수)가 아닌, Softmax 함수를 사용하게 된다. 이러한 이유로 Multiclass logistic regression을 Softmax Classifier라고도 부른다. Softmax function은 다음과 같다.

$$ Softmax(y_k) = S(y_k) = {e^{y_k} \over \sum_{j=1}^J e^{y_j}}$$

위에서 구한 $H_L(X)$에 Softmax function을 적용하여 transform해주면 다음과 같이 확률 값으로 해석할 수 있는 값을 반환해준다.

$$
\begin{align*}
Softmax(H_L(X)) =  
\begin{bmatrix}
\frac {e^{XW_A}} {e^{XW_A}+e^{XW_B}+e^{XW_C}} &
\frac {e^{XW_A}} {e^{XW_A}+e^{XW_B}+e^{XW_C}} &
\frac {e^{XW_A}} {e^{XW_A}+e^{XW_B}+e^{XW_C}}
\end{bmatrix}
\end{align*}
$$

이렇게 구한 확률 값들로 이루어진 행렬에서 가장 높은 확률을 가지는 Class를 찾고, 바로 그 Class가 Prediction의 결과가 된다. 이러한 알고리즘을 Softmax Classifier라고 한다.

<br>
<br>
### Cost

이제 남은 일은 어떻게 적절한 Weight $W_A$, $W_B$, $W_C$를 찾을 것인가에 대한 문제이다.

Binary class 문제에서는 loss function으로 다음과 같은 BCE(Binary Cross Entropy) loss를 사용하였다.

$$
\begin{align*}
BCE(p ,y) &=
-\sum_{i=1}^n \left \lbrack y_i log(p_i) + (1-y_i)log(1-p_i) \right \rbrack \\
&=-\sum_{i=1}^n \left \lbrack y_i log(\sigma(X_iW)) + (1-y_i)log(1-log(\sigma(X_iW)) \right \rbrack\\
&=-\sum_{i=1}^n \left \lbrack y_i log(\frac 1 {1+e^{-X_iW}}) + (1-y_i)log(1-\frac 1 {1+e^{-X_iW}}) \right \rbrack
\end{align*}
$$

<br>

반면, Multiclass 문제에서는 loss function으로 다음과 같은 CE(Cross Entropy) loss를 사용한다.


$$
\begin{align*}
CrossEntropy(S,L) &=
-\sum_{i=1}^n \sum_{j=1}^J \left \lbrack L_j log(S_j) \right \rbrack \\
&= -\sum_{i=1}^n \sum_{j=1}^J \left \lbrack L_j log(Softmax(X_iW)_ j) \right \rbrack \\
\end{align*}
$$

여기서 $L$ 행렬은 $y_i$에 대해 One-hot encoding 행렬로, class에 대해 Indicator function을 적용해준 행렬이라고 생각하면 된다. 예를들어 class의 수가 3개인 경우($J=3$), $L$은 다음과 같다.

$$
\begin{align*}
L &= [L_1\;L_2\;L_3] \\
&=
\begin{cases}
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix}
\;\;if\;y=1\\\\
\begin{bmatrix}
0 & 1 & 0
\end{bmatrix}
\;\;if\;y=2\\\\
\begin{bmatrix}
0 & 0 & 1
\end{bmatrix}
\;\;if\;y=3
\end{cases}
\end{align*}
$$

<br>
<br>
### BCE & CE

이 두 함수는 얼핏 보면 다르게 보이지만, 실제로는 같은 함수이다. 엄밀하게 말하면 BCE는 CE의 special case($J=2$)라고 할 수 있다. 이는 실제로 계산을 통해서도 확인할 수 있다.

$$
\begin{align*}
CrossEntropy(S,L) &=
-\sum_{i=1}^n \sum_{j=1}^2 \left \lbrack L_j log(S_j) \right \rbrack \\
&= -\sum_{i=1}^n \sum_{j=1}^2 \left \lbrack L_j log(Softmax(X_iW)_ j) \right \rbrack \\
&= -\sum_{i=1}^n \left \lbrack L_1 log(Softmax(X_iW)_ 1) + L_2 log(Softmax(X_iW)_ 2) \right \rbrack \\
&= -\sum_{i=1}^n \left \lbrack L_1 log(\frac {e^{X_iW_1}} {e^{X_iW_1}+e^{X_iW_2}}) + L_2 log(\frac {e^{X_iW_2}} {e^{X_iW_1}+e^{X_iW_2}}) \right \rbrack\\
&= -\sum_{i=1}^n \left \lbrack L_1 log(\frac 1 {1+e^{X_iW_2-X_iW_1}}) + L_2 log(1-\frac 1 {1+e^{X_iW_2-X_iW_1}}) \right \rbrack\\
&=-\sum_{i=1}^n \left \lbrack y_i log(\frac 1 {1+e^{-X_iW}}) + (1-y_i)log(1-\frac 1 {1+e^{-X_iW}}) \right \rbrack\\
&= BinaryCrossEntropy(p ,y)
\end{align*}
$$

다시 한 번 기억해야하는 것은 Binary logistic regression의 경우 sigmoid함수는 $y$가 1일 확률 값 하나를 반환하지만,

Multiclass logistic regression의 경우 softmax 함수는 class의 개수 만큼의 확률 값을 가지고있는 하나의 행렬을 반환한다는 사실이다. 즉, one-hot encoding된 $y$의 dimension 만큼의 행렬을 반환하게 된다.

pytorch를 활용한 multiclass logistic regression은 다음 포스트에서 살펴보도록 하겠다.

<br>
<br>
