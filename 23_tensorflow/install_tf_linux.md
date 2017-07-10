



# Install TF at Linux

* (1) install Anaconda (64bit)

my Anaconda python is python 3.6, which is still not supported by tf now.


* (2) create conda environment "envtf" with python 3.5

```
conda create --name envtf python=3.5
```


* (3) activate environment "envtf"

```
source activate envtf
```

now we come to python 3.5 environment. To deactivate this environment, use: `source deactivate envtf`


* (4) install tensorflow by cmd below

```
pip install --upgrade --ignore-installed  tensorflow
```

* (5) check 

```
(envtf) C:\Users\biny> python
Python 3.5.3 |Continuum Analytics, Inc.| (default, Feb 22 2017, 21:28:42) [MSC v.1900
] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
>>>
```

