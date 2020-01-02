---
layout: post
title: Linear Regression
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.

<br>
<br>
### Linear Regression in Statistics

우선, Simplicity를 위하여 변수가 1개인 Simple Linear Regression을 다루도록 하겠다.

우리는 다음과 같이 $x$와 $y$ 간의 선형 관계를 가정하고, 그러한 관계에 대한 정보를 가지고 있는 모수, 즉 미지의 값 $\beta_0$와 $\beta_1$이 존재한다고 가정한다.

$$ y_i = \beta_0 + \beta_1 x_i + \epsilon_i $$

하지만 현실에서는 이러한 값을 알지 못하고, 우리는 주어진 데이터 $x$로부터 $\beta_0$와 $\beta_1$을 추정하고자 한다. 우리가 추정하고자 하는 모수 $\beta_0$, $\beta_1$에 대한 추정치를 $\hat \beta_0$, $\hat \beta_1$ 이라고 한다.

$$ \hat y_i = \hat \beta_0 + \hat \beta_1x_i $$

$\beta_0$와 $\beta_1$을 추정하는 방법에 대한 이야기는 지난 [Linear regression 포스트](https://kjhov195.github.io/2019-10-26-linear_regression/)에서도 다루었으니 간략하게만 설명하도록 하겠다.

$$Q(\beta_0,\beta_1) = \sum\limits_{n=1}^{n} \epsilon^{2} = \sum\limits_{n=1}^{n} (y_i-\beta_0-\beta_1x_i)^{2}$$

$$Argmin_{\beta_0, \beta_1}\;Q(\beta_0,\beta_1) \vert _ {\beta_0 = \hat \beta_0, \beta_1 = \hat \beta_1}$$

우리는 위 식을 풀어 $y$와 $\hat y$의 차이에 대한 제곱합(Error sum of squares)을 가장 작게 만들어주는 $\beta_0$와 $\beta_1$을 찾을 수 있고, 이를 Least Squared Estimator for $\beta_0$, $\beta_1$, 혹은 $\hat \beta_{0,LSE}$, $\hat \beta_{1,LSE}$라고 한다.



<br>
<br>
### Linear Regression in Machine Learning

Computer Science, Machine Learning 분야의 linear regression에서도 마찬가지로 $y$와 $\hat y$의 차이에 대한 제곱합을 최소화시키는 모수들을 찾고자 한다.

다만 두 분야에서 사용하는 Notation이 조금 다르다. Notation만 조금 달라질 뿐 개념은 완벽하게 같으므로 새롭게 공부할 내용은 없지만, 머신러닝/딥러닝 분야의 논문을 읽을 때 거의 항상 $H(x)$, $W$, $b$, $cost(W,b)$가 등장하므로 익숙해질 필요는 있다.

Machine learning 분야에서 통용되는 표기로는 $\hat y$가 아닌 Hypothesis function $H(x)$을 사용한다.

모수들 또한 $\beta$가 아닌 $W$(weight)와 $b$(bias)로 나타내며, Error sum of squares 대신, sum of squares를 n으로 나눈 Cost라는 개념을 사용한다. 유일하게 다른 점은 Minimize하고자 하는 목적 함수인데, Error sum of squares를 사용하든 Cost를 사용하든 결국 차이는 곱해진 $1/n$인데, 이는 미분 과정에서 상수로 취급되므로 모수에 대한 추정값은 이와 상관없이 같다.

$$ H(x) = Wx + b $$

$$ cost(W, b) = \frac{1}{n} \sum^n_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 $$

정리하면 $H(x)$는 주어진 $x$로 예측된 $y$에 대한 추정치이며, $cost(W, b)$는 $H(x)$ 가 $y$ 를 얼마나 잘 예측했는가를 나타내는 지표로 볼 수 있다. 우리는 이 Cost를 최소화시키는 $W$와 $b$를 찾게 될 것이고, 이렇게 찾은 $W$와 $b$를 통하여 $H(x)$에 대한 Regression 모델링이 가능하다.


|  <center>Statistics </center> |  <center>Machine Learning</center> |  
|:--------|:--------:|--------:|
| <center> $\hat y$ </center> | <center> $H(x)$ </center> |
| <center>  $\beta_0$ </center> | <center> $b$ </center> |
| <center>  $\beta_1$ </center> | <center> $W$ </center> |
| <center>  MSE </center> | <center> $Cost$ </center> |


<br>
<br>
### Source Code

```
### Linear Regression

# Dataset
X_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# Weight Initialization
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Setting Optimizer
optimizer = torch.optim.SGD([W, b], lr=0.01)

# Training
n_epochs = 2000
for epoch in range(n_epochs + 1):
    # H(x)
    hypothesis = X_train * W + b
    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)
    # Updating W, b
    optimizer.zero_grad()  # Initialize Gradients calculated in the last step.
    cost.backward()        # Getting new Gradients based on Differentiating of this step.
    optimizer.step()       # Update W, b with new Gradients(& learning_rate)

    # print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, n_epochs, W.item(), b.item(), cost.item()
        ))
```

Full code는 위와 같다. Line by line으로 조금 더 자세히 살펴보도록 하자.

<br>
<br>
### Line by line

```
# Dataset
X_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
```
_torch.Size([3, 1])_ 의 shape을 가진 Training set(X_train과 y_train)을 정의한다.

<br>

```
# Weight Initialization
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

Training 전에 W와 b Tensor를 0으로 initialize해 준다.

__torch.zeros()__ 함수의 첫 번째 인자는 tensor의 __shape__ 을 의미하며, 1이므로 _torch.Size([1])_ 의 shape을 가진 tensor W, b를 정의해 준 것이다.

__torch.zeros()__ 함수의 옵션 __requires_grad=True__ 는 해당 tensor에 대한 gradient를 계산하겠다는 의미이며, 즉 해당 Tensor가 우리가 추정하고자 하는 모수에 대한 Tensor임을 명시해주는 옵션이다. 해당 옵션이 __True__ 일 경우, 미분을 통하여 Gradient를 계산하고 training 과정에서 epoch마다 update가 이루어 진다.

<br>

```
# Setting Optimizer
optimizer = torch.optim.SGD([W, b], lr=0.01)
```

다양한 Optimizer를 정의할 수 있는데, Stochastic Gradient Descent Optimizer를 사용하기로 하였다. SGD에 대한 자세한 내용은 다음 포스트에서 설명하도록 하겠다.

<br>

```
# Training
n_epochs = 2000
for epoch in range(n_epochs + 1):
    # H(x)
    hypothesis = X_train * W + b
    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)
    # Updating W, b
    optimizer.zero_grad()  # Initialize Gradients calculated in the last step.
    cost.backward()        # Getting new Gradients based on Differentiating of this step.
    optimizer.step()       # Update W, b with new Gradients(& learning_rate)

    # print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, n_epochs, W.item(), b.item(), cost.item()
        ))
```

SGD optimizer의 경우 batch size는 1이며, epoch의 수는 2000으로 설정해주었다. 각 epoch에서  $Hypothesis$를 계산하고, 해당 $Hypothesis$와 True $y$의 오차를 계산, cost를 계산하게 된다.

해당 cost를 $W$와 $b$에 대하여 미분하기 전에, 직전 epoch에서 계산하였던 미분 값을 0으로 초기화시키고,(optimizer.zero_grad())

해당 cost를 $W$와 $b$에 대하여 미분하여 Gradient를 계산한 다음,(cost.backward())

계산된 Gradient를 반영하여 $W$와 $b$를 새로운 값으로 Update하여 준다.(optimizer.step())

<br>

```
# Updating W, b
optimizer.zero_grad()  # Initialize Gradients calculated in the last step.
cost.backward()        # Getting new Gradients based on Differentiating of this step.
optimizer.step()       # Update W, b with new Gradients(& learning_rate)
```

code를 이해하면 알 수 있듯, 자연스럽게 위 세 함수는 거의 항상 함께 사용되므로 익숙해질 필요가 있다.

<br>
<br>
### Reference
[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
