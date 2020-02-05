---
layout: post
title: CUDA/CuDNN
subtitle: Deep Learning
category: Deep Learning
use_math: true
---


<br>
<br>
### CUDA

```
# installing CUDA
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-9-0

# check
nvidia-smi
```



<br>
<br>
### cuDNN

```
# installingcuDNN
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install libcudnn7-dev

# check
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
