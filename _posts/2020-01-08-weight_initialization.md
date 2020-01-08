---
layout: post
title: Initialization-Xavier/He
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 Stanford University School of Engineering의 [CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7) 강의자료와 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.


<br>
<br>

### Initialization

Neural Networks를 학습시킬 때 Weight을 어떻게 초기화할 것인지는 아주 중요한 inssue이다.

만약 모든 weights를 0으로 초기화한다면 어떤 일이 생길지 생각해보자.

__같은 output 값을 얻음__ $\rightarrow$ __같은 gradient 값을 얻음__ $\rightarrow$ __같은 weight 을 업데이트함__


즉, 모든 뉴런들이 동일한 연산을 수행하게 되고, back propagation 또한 동일한 gradient를 계산하게 될 것이다.

$$\text{All the neurons would do the same thing.}$$



<br>
<br>
### Xaiver initialization

##### Xavier normal initialization

$$
\begin{align*}
W \sim N(0, \sqrt{\frac 2 {n_{in}+n_{out}})}\\
\end{align*}
$$

<br>
##### Xavier normal initialization

$$
\begin{align*}
W \sim Unif(-\sqrt{\frac 6 {n_{in}+n_{out}}},\;\sqrt{\frac 6 {n_{in}+n_{out}}})
\end{align*}
$$


<br>
<br>
### He initialization

Xaiver Initialization의 변형이다. Activation Function으로 ReLU를 사용하고, Xavier Initialization을 해줄 경우 weights의 분포가 대부분이 0이 되어버리는 Collapsing 현상이 일어난다. 이러한 문제점을 해결하는 방법으로 He initialization(Xaiver with $1 \over 2$) 방법이 고안되었다.


##### He normal initialization

$$
\begin{align*}
W \sim N(0, \sqrt{\frac 2 {n_{in}})}\\
\end{align*}
$$

<br>
##### He normal initialization

$$
\begin{align*}
W \sim Unif(-\sqrt{\frac 6 {n_{in}}},\;\sqrt{\frac 6 {n_{in}}})
\end{align*}
$$


<br>
<br>
### Default of ```torch.nn.Linear```

만약 Pytorch에서 ```torch.nn.Linear()``` 함수를 사용할 때 따로 Weight initialization을 설정해주지 않으면 weights가 어떻게 initialized될까?

[Pytorch의 git](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py)에서 __torch.nn.Linear()__ 함수에 해당하는 부분을 찾아보았다. Weight initialization이 없을 때 어떻게 weight을 초기화할 것인지에 대한 부분도 아래와 같이 명시되어 있다.

<br>

```
def reset_parameters(self):
    bound = 1 / math.sqrt(self.weight.size(1))
    init.uniform_(self.weight, -bound, bound)
    if self.bias is not None:
        init.uniform_(self.bias, -bound, bound)
```

$$ W \sim Unif(-{1 \over \sqrt{n_{in}}},{1 \over \sqrt{n_{in}}})$$

명시하지 않더라도 위와 같이 initialize해 주도록 설정되어 있다. 단, 이러한 default initialization은 __torch.nn.Linear()__ layer에만 해당하며, 각 layer의 종류마다 다른 dafault initialization 방법을 선택한다. 예를들어 Conv2d layer의 경우 따로 initialization 방법을 정해주지 않을 경우 Xavier initialization 방법을 사용한다.

<br>
<br>
### Example


<br>

```
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

# setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# nn layers
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()
# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning finished')
```

[앞선 포스트](https://kjhov195.github.io/2020-01-07-activation_function_2/)와 같은 구조의 모델을 사용하되, weight initialization 방법만 ```xavier_uniform_()```으로 바꿔주었다.

<br>
```
# Test the model using test sets
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
```

[Softmax Classifier](https://kjhov195.github.io/2020-01-05-softmax_classifier_3/) with 1 layer, default initialization, sigmoid activation $\rightarrow$ Accuracy 86.9%

[Neural Networks](https://kjhov195.github.io/2020-01-07-activation_function_2/) with 3 layer, ```torch.nn.init.normal_``` initialization, ReLU activation $\rightarrow$ Accuracy 94.7%

Neural Networks with 3 layers, ```torch.nn.init.xavier_uniform_``` initialization, ReLU activations $\rightarrow$ Accuracy 98.1%

이번 예제의 모델에서는 98.1%의 Accuracy를 보여준다. Initialization 방법만 바꿔주었을 뿐인데도 매우 큰 폭으로 성능이 향상되는 것을 확인할 수 있다.

<br>
<br>
### Reference

[CS231n](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk), Stanford University School of Engineering

[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
