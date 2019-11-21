---
layout: post
title: Nonparametric_hw3
subtitle: Machine Learning
category: ML
use_math: true
---


<br>
<br>

# <center>Nonparametric Statistics</center>
##### <center>2018321086 Jaehun Kang</center>

<br>
### <center>Short Article</center>
### <center> Statistical Learning : Ensemble in ML</center>

<br>

분류 문제를 해결할 수 있는 모델은 무수히 많이 존재한다.(Logistic regression, Random Forest 등...) 어떠한 Classifier $f$를 생각해보자.

<br>
### 1. Variance in classification

일반적인 Classification 방법을 사용한다면 데이터셋이 달라짐에 따라 classifier $f$는 아주 크게 변할 수 있다. ___Variance___ 는 Training dataset에 대한 Classifier $f$의 __민감도(sensitivity)__ 를 측정하는 척도로 볼 수 있다. 분류 모델의 Variance는 다양한 요인에 의해 영향을 받는다.

<br>
##### 1.1. Variance $\propto$ 1/N

일반적으로 모델의 Variance는 Training dataset의 크기 N과 반비례하는 성향을 보인다. N(Training dataset의 크기)이 커진다면 그만큼 모델의 Variance는 줄어들 것이다.

<br>
##### 1.2. Variance $\propto$ Complexity of model

반면, 모델의 Variance는 모델의 복잡한 정도(Complexity)와 비례하는 성향을 가진다. 즉, 일반적으로 모델이 복잡하거나 Overfitted될수록 Variance는 커진다.

<br>
### 2. Bias in classification

##### 2.1. Bias $\propto$ Simplicity of model

어떠한 모델 $f$를 선택하더라도, 모델의 구조가 너무 간단하면 모델이 잘 data에 fitting되지 않아 bias가 증가할 수 있다.

<br>
### 3. Stable vs Unstable

상대적으로 Low Variance, High Bias의 모델을 우리는 __Stable__ 하다고 한다. 예를 들어, Logistic Regression 등이 있다.

상대적으로 High Variance, Low Bias의 모델을 우리는 __Unstable__ 하다고 한다. 예를 들어, Decision Tree, Neural Networks 등이 있다.

만약 모델에 __불안정성(Unstablity)__ 이 존재한다면 Ensemble을 통한 모델 성능 향상의 핵심 재료로 활용할 수 있다. Low Bias를 가지는 모델의 High Variance 문제만 해결할 수 있다면 기존의 모델보다 훨씬 더 좋은 모델을 만들 수 있기 때문이다.

<br>
### 4. How?

<br>
##### 4.1. Bagging

Bagging은 복원추출을 통하여 dataset을 바꿔가며 classification을 하고, 그 결과들에 대하여 __Equal weight__ 을 주어 투표하여 가장 높은 표를 얻은 Class를 predicted class로 선택하는 것이다.

다음은 간단한 Bagging의 과정에 대한 설명이다.

---

우리는 J개의 Class를 예측할 수 있는 Classifier를 만들고자 한다.

<br>

1. Bootstrap을 통하여 B개의 training set을 sampling한다.(sample의 수가 N이라면, 복원추출로 N개 Sampling을 총 B번 한다.)

$$T^{(1)}, T^{(2)},  ..., T^{(B)}$$

<br>

2. 이렇게 만들어 낸 B개의 training set을 사용하여 B개의 Classifier를 만들어낸다.

$$C(x,T^{(1)}), C(x,T^{(2)}),...,C(x,T^{(B)})$$

<br>

3. 우리는 B개의 Classifier를 만들어냈다. 이 B개의 Classifier들은 어떠한 Input에 대하여 서로 다른 예측을 할 수 있다. B개의 Classification 결과, 우리는 _(총 J개의 Classes 중)_ j번째 class로 예측된 횟수를 $N_j$라고 할 수 있다.

$$ N_j = \sum_{b=1}^{B} I[C(x,T^{(b)})=j] \;\;\;\;for\;j=1,2,...,J$$

<br>

4. 이렇게 B번의 classification을 하였을 때 가장 많이 예측된 결과를 찾는다. 이를 최종 predicted class로 선택한다.

$$C_B(x) = argmax_j {(N_j)}$$

---

정리하면 Bagging은 __unweighted majority voting__ 을 사용한다고 할 수 있다. 이러한 Bagging을 통해 unstable한 model의 variance를 줄일 수 있다.

Bagging의 대표적인 예시로는 Random forests가 있다.


<br>
##### 4.2. Boosting

Boosting은 training을 여러 차례 반복하면서 직전의 model에서 잘못 분류된 sample을 더 잘 맞출 수 있도록 개선해 나가는 과정이라고 할 수 있다. 기본적인 아이디어는 다음과 같다.

---

첫 번째 classifier $f_1$을 만든 뒤, 여기서 발생하는 errors를 correct하는 $f_2$를 학습시킨다.

두 번째 classifier $f_2$를 만든 뒤, 역시 여기서 발생하는 errors를 correct하는 $f_3$를 학습시킨다.

이하 이 과정을 계속 반복하게 된다.

---



AdaBoost, Gradient Boost, XGBoost 등이 Boosting의 대표적인 예시다.

|  <center> Model </center> |  <center> Property </center> |  <center> Paper </center> |  
|:--------|:--------:|--------:|
| <center> AdaBoost(1996) </center> | <center> 오답에 가중치 부여 <br> 다수결을 통한 정답 분류 </center> | [Experiments with a New Boosting Algorithm](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)|
| <center> GradientBoost(1999) <br>(GBM) </center> | <center> Loss function 수정. <br> Gradient를 활용하여 오답에 가중치 부여 </center> | [GREEDY FUNCTION APPROXIMATION: A GRADIENT BOOSTING MACHINE](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451)|
| <center> XGBoost(2014) </center> | <center> GradientBoost 개선 <br> CPU, Memory 관리를 통한 효율적인 computing </center> | [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)|
| <center> LightGBM(2016) </center> | <center> GradientBoost 개선 <br> XGBoost에 비하여 더 효율적인 computing <br> 대용량 데이터 학습 가능 <br> Approximatesx the split 사용 </center> | [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) |

<br>

이 중, AdaBoost에 대해 간단히 살펴보도록 하자. AdaBoost는 잘못 분류된 observation에 더 큰 weight을 주게 된다. 즉, training data에 unequal weights을 주는 것이다. 이러한 Boosting을 통해 우리는 Variance와 Bias 모두 줄일 수 있게 된다.

---

[AdaBoost]

우리의 데이터셋에 데이터가 $N$개가 있으며, 2개의 class를 classification하고자 한다.

처음에는 데이터들에 대한 가중치가 각각 $1 \over N$일 것이다.($w_1=w_2=\cdots=w_N = {1 \over N}$)

<br>

(a) Dataset with $w_i(i=1,2,\cdots,N)$를 통해 Classifier $C_m(X)$를 학습시킨다.

(b) $err_m$을 다음과 같이 계산한다.

$$err_m = {\sum_{i=1}^N w_i I(y_i \neq C_m(x_i))  \over {\sum_{i=1}^N w_i}}$$

(c) $\alpha_m$ 을 다음과 같이 계산한다.

$$\alpha_m = log{(1-err_m) \over err_m}$$

(d) $w_i(i=1,2,\cdots,N)$을 다음과 같이 update한다.

$$w_{i,(update)} = w_{i,(previous)} \cdot \exp[\alpha_m \cdot I(y_i \neq C_m(x_i))] $$

<br>

그리고 (a)~(d) 과정을 반복한다.($m = 1,2, \cdots, M$)

최종 Output은 다음과 같다.

$$C_{Adaboost}(x) = sign[\sum_{m=1}^M \alpha_m C_m(x)]$$

<br>
##### 4.3. Stacking

Stacking은 Stacked Generalization이라고도 불린다. Stacking은 다른 여러 알고리즘의 예측 결과를 결합하여 새로운 알고리즘을 학습하는 과정을 의미한다. 학습 과정을 간단하게 설명하면 다음과 같이 2-stage로 나눌 수 있다.

---

(stage 1) Data를 통해 각각의 다른 알고리즘들을 학습한다.

(stage 2) stage 1에서 만든 알고리즘들의 예측 결과를 추가적인 input으로 사용하여 __Combiner algorithm__ (혹은 meta classifier)을 만들고, 이를 통하여 final prediction을 구한다.

---

사실 이렇게 설명하면 이해하기가 쉽지 않다. 이해하기 쉽도록 _Bhavesh Bhat(2019), Stacking Classifier | Ensemble Classifiers | Machine Learning_ 을 참고하여 전체적인 과정을 직접 그려보았다.

<br>

<center><img src = '/post_img/191120/image1.png'/></center>

<br>

기본적인 stacking의 아이디어를 도식화한 그림이다. 단계별로 살펴보도록 하겠다.

<br>

(1) 데이터셋 나누기

<br>

<center><img src = '/post_img/191120/image2.png'/></center>

<br>

우리에게 10000x4의 Training data($x$)와 10000x1의 Target data($y$)가 주어졌다고 가정하자.

우선, 일반적으로 7:3의 비율로 dataset을 나눌 수 있으로 7000x4의 training set과 3000x4의 test set으로 나누게 된다. 이 training set을 다시 한 번 training set1과 training set2로 나눌 수 있는데, training set1은 1-level classifier를 학습시킬 때 사용할 데이터이고, training set2는 최종 모델인 meta classifier를 학습시킬 때 사용할 데이터가 된다.

<br>
(2) 1-level-classifiers 학습

<br>

<center><img src = '/post_img/191120/image3.png'/></center>

<br>

앞서 나눈 Training set 중, Training set1을 통하여 여러 모델을 학습시킨다. 이 예시에서는 세 가지 classification models를 학습시켰다.


<br>
(3) meta-classifier의 training data 만들기

<br>

<center><img src = '/post_img/191120/image4.png'/></center>

<br>

앞서 만든 여러 모델들에 Training set2를 input으로 주었을 때, class에 대한 예측 확률($\hat p$)을 얻을 수 있다. 이렇게 각 model마다 해당 class에 대한 확률 값을 얻을 수 있을 것이다. 이렇게 세 모델에서 얻은 예측 확률들로 2000x3의 dimension을 가진 [p1, p2, p3] 행렬을 만든다.


<br>
(4) meta-classifier 학습

<br>

<center><img src = '/post_img/191120/image5.png'/></center>

<br>

이렇게 만든 [p1, p2, p3] 행렬을 meta classifier의 training dataset으로 사용하여 최종 모델을 학습시킨다.

<br>
### 5. More




<br>
<br>
### Reference
김현중(2018), Data Mining, 연세대학교

Bhavesh Bhat(2019), [Stacking Classifier | Ensemble Classifiers | Machine Learning](https://www.youtube.com/watch?v=sBrQnqwMpvA)
<br>
<br>
