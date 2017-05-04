# Install TF at Win

* (1) install Anaconda (64bit)

my Anaconda python is python 3.6, which is still not supported by tf now.


* (2) create conda environment "envtf" with python 3.5

```
conda create --name envtf python=3.5
```


* (3) activate environment "envtf"

```
activate envtf
```

now we come to python 3.5 environment. To deactivate this environment, use: `deactivate envtf`


* (4) install tensorflow by cmd below

```
pip install --upgrade tensorflow
```

get error

```
FileNotFoundError: [WinError 2] The system cannot find the file specified:  'C:\\Users\\biny\\Anaconda3\\envs\\envtf\\li
b\\site-packages\\setuptools-27.2.0-py3.5.egg'
```

then (5)


* (5) 

```
(envtf) C:\Users\biny> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl
```

* (6) check 

```
(envtf) C:\Users\biny> python
Python 3.5.3 |Continuum Analytics, Inc.| (default, Feb 22 2017, 21:28:42) [MSC v.1900
] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
>>>
```

