---
layout: post
title: Pytorch Dataset/Dataloader
subtitle: Deep Learning
category: Deep Learning
use_math: true
---

<br>

우선, 해당 포스트는 [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)의 자료를 기본으로 하여 정리한 내용임을 밝힙니다.

<br>
<br>
### Pytorch Dataset

Pytorch의 ```torch.utils.data.Dataset```은 Pytorch에서 제공하는 모듈로써 이 모듈을 상속하여 새로운 class를 만듦으로써 우리가 원하는 custom dataset을 지정해줄 수 있게 된다.

custom dataset을 만들 때, 이 데이터셋의 총 데이터 수를 반환하는  ```__len__``` method와, 주어진 index에 대응되는 데이터를 반환하는 ```__getitem__``` method라는 두 가지 method를 만들게 된다.

<br>
<br>
### Pytorch Dataloader

먼저 ```torch.utils.data.Dataset```를 만들고 나면, ```torch.utils.data.Dataloader```를 사용할 수 있게 된다. ```torch.utils.data.Dataloader```는 Pytorch에서 제공하는 모듈로써 두 가지 옵션을 지정해주어야 한다.

첫 번째 옵션은 ```batch_size```이며, Dataset으로부터 불러올 각 minibatch의 크기를 의미한다. 통상적으로 2의 제곱수로 설정하는 경우가 많다.(16, 32, 64, ...)

두 번째 옵션은 ```shuffle```이다. shuffle은 매 Epoch마다 dataset의 순서를 섞어서 데이터가 학습되는 순서를 바꿔주도록 해준다. 이 옵션을 주게 되면 우리의 모형이 데이터셋의 순서를 외우지 못하게하여 overfitting을 방지해줄 수 있다.

<br>
<br>
### Source code

```
import torch
import torch.utils.data
import numpy as np

# dataset
X_train = np.array([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = np.array([[152], [185], [180], [196], [142]])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

dataset = CustomDataset()
dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
dataloader
```

<br>
<br>
### Example

다음은 이번 포스트에서 배운 ```torch.utils.data.Dataset```과 ```torch.utils.data.Dataloader```의 활용을 [Multivariate Linear Regression](https://kjhov195.github.io/2020-01-03-multivariate_linear_regression/)의 예제에 적용한 것이다.

```
import torch
import torch.utils.data
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

class MultivariateLinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# dataset
X_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# weight initialization
model = MultivariateLinearRegressionModel()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

n_epochs = 20
for epoch in range(n_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        X_train, y_train = samples

        # H(x)
        prediction = model(X_train)

        # cost
        cost = torch.nn.functional.mse_loss(prediction, y_train)

        # updating weights
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

model.state_dict()
```


<br>
<br>
### Reference
[모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)
