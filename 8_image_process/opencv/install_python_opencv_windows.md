# Environment Setup

* python 2.7 virtual env
* download opencv-3.1.0.exe from http://opencv.org/releases.html
* extract opencv-3.1.0
* copy opencv-3.1.0\build\python\2.7\x64\cv2.pyd to D:\Projects\py2env\Lib\site-packages
* import the package at py code
```python
import cv2
```


# Issue fixing

* Error: dll load failed
   * Fixing: Install Visual C++ 2015 redistribution package


# References

* python example
   * opencv-3.1.0\sources\samples\python
* run demo

```
(py2env) C:\mine\tools\opencv-3.1.0\sources\samples\python>python demo.py
```


