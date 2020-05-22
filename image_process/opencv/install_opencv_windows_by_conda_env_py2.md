# How to install OpenCV by conda py2 environment at Windows 64


* Download conda 4.3.1 (python3, win-64) at https://repo.continuum.io/archive/Anaconda3-4.3.1-Windows-x86_64.exe

* Install conda

* create python2.7 environment by conda : `conda create --name envopnecv python=2.7`

* `activate envopnecv`

* `conda install scikit-learn`

* check python exe path
```
>>> import sys
>>> sys.executable
'C:\\Users\\biny\\Anaconda3\\envs\\envopnecv\\python.exe'
```

* download opencv-3.1.0.exe from http://opencv.org/releases.html

* extract  opencv-3.1.0

* copy opencv-3.1.0\build\python\2.7\x64\cv2.pyd to C:\Users\biny\Anaconda3\envs\envopnecv\Lib\site-packages

* pip install matplotlib
