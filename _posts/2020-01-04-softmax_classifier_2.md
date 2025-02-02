---
layout: post
title: Softmax Classifier(2)
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>
### Example 1.

앞선 [post](https://kjhov195.github.io/2020-01-04-softmax_classifier_1/)에서 Multiclass logistic regression에 대해 살펴보았다.

이번에는 Pytorch를 활용하여 실제로 classification 문제를 풀어보도록 하자.

<br>
```
import torch

# For reproducibility
torch.manual_seed(1)

# training set
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# weight initialization
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = torch.optim.Adam([W, b], lr=0.1)

n_epochs = 4000
for epoch in range(n_epochs + 1):
    # Hypothesis
    hypothesis = torch.nn.functional.softmax(x_train.matmul(W) + b, dim=1)

    # cost
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

    cost = -(y_one_hot*torch.log(hypothesis)).sum(dim=1).mean()

    # updating weights
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))
```

역시 Linear regression의 pytorch code와 크게 다르지 않다. 새롭게 바뀐 부분을 중심들로 살펴보도록 하자.

<br>

```
# weight initialization
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

우선, $W$ 행렬의 shape은 $4 \times 3$이다. 그 이유는 training data $X$에서 feature의 개수가 4개이며, 우리가 classification하고자 하는 $y$의 class 수가 3개이기 때문이다.

<br>

```
# Hypothesis
hypothesis = torch.nn.functional.softmax(x_train.matmul(W) + b, dim=1)
```

Pytorch의 함수 ```torch.nn.functional.softmax()``` 를 사용하여 hypothesis를 계산해 주었다. 즉, hypothesis 객체는 다음을 계산한 결과이다.

$$
\begin{align*}
\text{hypothesis} &= Softmax(H_L(X)) \\
&=  
\begin{bmatrix}
\frac {e^{X_1W_A}} {e^{X_1W_A}+e^{X_1W_B}+e^{X_1W_C}} &
\frac {e^{X_1W_B}} {e^{X_1W_A}+e^{X_1W_B}+e^{X_1W_C}} &
\frac {e^{X_1W_C}} {e^{X_1W_A}+e^{X_1W_B}+e^{X_1W_C}}\\
\vdots & \vdots & \vdots\\
\frac {e^{X_8W_A}} {e^{X_8W_A}+e^{X_8W_B}+e^{X_8W_C}} &
\frac {e^{X_8W_B}} {e^{X_8W_A}+e^{X_8W_B}+e^{X_8W_C}} &
\frac {e^{X_8W_C}} {e^{X_8W_A}+e^{X_8W_B}+e^{X_8W_C}}
\end{bmatrix}_ {8 \times 3}
\end{align*}
$$

<br>

```
# cost
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

cost = -(y_one_hot*torch.log(hypothesis)).sum(dim=1).mean()
```

이 부분에서는 ```torch.scatter_```와 ```torch.unsqueeze``` 함수를 활용하여 $y$를 one-hot encoding해주었다.

우선, ```torch.zeros_like(hypothesis)```에서 hypothesis와 같은 shape의 0 tensor를 만들어 준다. 여기서 hypothesis는 _torch.Size([8, 3])_ 의 shape이었으므로, y_one_hot의 shape 또한 _torch.Size([8, 3])_ 이 된다.

그 후에 ```y_train.unsqueeze(1)```를 통하여 _torch.Size([8])_ 의 shape을 가진 y_train을 _torch.Size([8, 1])_ 의 shape으로 reshape해 준다.

마지막으로 ```scatter_(1, y_train.unsqueeze(1), 1)```를 통하여 one-hot encoding을 해준다. scatter 함수 인자들의 의미는 순서대로 __dimension=1__ 에 해당하는 방향, 즉 열 방향으로 __y_train.unsqueeze(1)__ 의 값들을 흩뿌려주는데, __1__ 이라는 숫자를 사용하여 흩뿌리라는 의미이다.

이후, ```cost = -(y_one_hot*torch.log(hypothesis)).sum(dim=1).mean()```로써 Cross Entropy loss를 활용하여 cost를 계산해 주었다.

<br>

```
y_pred = torch.argmax(torch.softmax(x_train.matmul(W),dim=1),dim=1)
print(torch.equal(y_train,y_pred))
```

추정된 $W$로 training data에 대하여 다시 예측해보았을 때 모든 데이터에 대하여 class를 정확하게 맞추고 있는 것을 확인할 수 있다.

<br>
<br>
### Using High-level API

```
import torch

class SoftmaxClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# training set
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

n_epochs = 4000
for epoch in range(n_epochs + 1):
    # Hypothesis
    linear = model(x_train)

    # cost
    cost = torch.nn.functional.cross_entropy(linear, y_train)

    # Updating weights
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))
```

여기서 매우 중요한 포인트는 Pytorch에서 loss function으로 __Cross Entropy loss__ 를 사용할 경우, 해당 함수 내에서 자동으로 softmax 값을 계산해준다는 점이다. 즉, 우리는 다음 두 가지 사항을 명심해야한다.

우선, ```SoftmaxClassifierModel()``` Class를 정의할 때 Softmax 함수 값을 return해주는 것이 아닌 선형 함수 $XW+b$ 를 return하도록 만들어 주어야 한다.

또한, ```torch.nn.functional.cross_entropy()```를 사용할 때에는 인자로 $XW+b$와 $Y$(one-hot encoding이 되지 않은)를 사용해야 한다는 점을 잊지 말아야 한다.

<br>
<br>
### Example 2. Real dataset

이번에는 실제 데이터에 적용하여 Multiclass logistic regression으로 classification 문제를 풀어보도록 하자. 데이터는 __pid.dat__ 와 __pidtest.dat__ 데이터로, [여기](https://github.com/kjhov195/data_mining/tree/master/2_data)에 업로드 해놓았다.

<br>
```
import torch
import pandas as pd
import numpy as np

# For reproducibility
torch.manual_seed(1)

# model
class SoftmaxClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 2)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# dataset
# training set
df_train = pd.read_csv("pid.dat",sep=',',header=None)
x_train = np.array(df_train.iloc[:,:7])
y_train = np.array(df_train.iloc[:,7])

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
y_train[y_train==1] = 0
y_train[y_train==2] = 1

# test set
df_test = pd.read_csv("pidtest.dat",sep=',',header=None)
x_test = np.array(df_test.iloc[:,:7])
y_test = np.array(df_test.iloc[:,7])

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)
y_test[y_test==1] = 0
y_test[y_test==2] = 1

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

n_epochs = 10000
for epoch in range(n_epochs + 1):
    # Hypothesis
    XW = model(x_train)

    # cost
    cost = torch.nn.functional.cross_entropy(XW, y_train)

    # Updating weights
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))

# validation
y_pred = torch.argmax(model(x_train),dim=1)
(y_train == y_pred).numpy().mean() #accuracy

# test
y_pred = torch.argmax(model(x_test),dim=1)
(y_test == y_pred).numpy().mean() #accuracy
```


<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
