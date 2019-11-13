---
layout: post
title: Spark install(1)-Anaconda,Jupyter
subtitle: Spark
category: Ubuntu, Anaconda, Jupyter notebook
use_math: true
---

<br>
<br>
### Ubuntu
Ubuntu는 Ubuntu 18.04 LTS Minimal를 사용한다.

```
$ cd $HOME
$ mkdir downloads
$ sudo apt-get update
$ sudo apt-get install nano
```

<br>
<br>
### Anaconda

#### install

https://repo.continuum.io/archive/ 에서 Anaconda의 최신 버전을 확인한다. 2019.11.13. 기준, __Anaconda3-2019.10-Linux-x86_64.sh__ 이 최신 버전이다.

```
$ cd ~/downloads
$ wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
$ bash Anaconda3-2019.10-Linux-x86_64.sh
```

이제 다음의 질문들에는 다음과 같이 입력하면 된다.

```
Welcome to Anaconda3 2019.10
In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
```
Enter


```
Do you accept the license terms? [yes|no]
```
yes


```
Anaconda3 will now be installed into this location:
/home/kjhov195/anaconda3
  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below
```
Enter

```
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
```
no

<br>
#### path

```
$ nano ~/.bash_profile
```

다음을 입력한다. 주의할 점은 GCP를 사용할 경우 사용할 경우, 사용자 이름이 ubuntu로 되어있지 않고, 본인의 아이디로 되어있으므로 경로를 잘 수정해주어야 한다.

```
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

## User specific environment and startup programs

# Anaconda
export ANACONDA_HOME="/home/kjhov195/anaconda3"
export PATH=${ANACONDA_HOME}/bin:$PATH
```

bash_profile 수정 후, 바뀐 환경변수를 적용해준다.

```
$ source ~/.bash_profile
```

<br>
<br>
### Jupyter

```
$ cd $HOME
$ jupyter notebook --generate-config

$ jupyter notebook password

$ mkdir ~/certs
$ cd ~/certs
$ sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

$ cd ~
$ nano ~/.jupyter/jupyter_notebook_config.py
```

아래의 내용을 추가하자. 사용하고자 하는 포트를 설정해준다. 10001번 포트를 사용하자.

```
c = get_config()

# Notebook config this is where you saved your pem cert
c.NotebookApp.certfile = u'/home/kjhov195/certs/mycert.pem'
c.NotebookApp.keyfile = u'/home/kjhov195/certs/mykey.key'

# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
# Don't open browser by default
c.NotebookApp.open_browser = False
# Fix port to 10001
c.NotebookApp.port = 10001
```

Jupyter notebook을 실행해보자.

```
$ jupyter notebook
```

이제 __https://외부IP:포트__ 로 접속할 수 있다.

<br>
<br>
