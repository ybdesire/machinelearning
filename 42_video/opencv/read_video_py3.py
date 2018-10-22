#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

cap = cv2.VideoCapture("xxx.mp4")

while(1):
    #ret和frame都是返回值，后者代表帧数
    ret,frame = cap.read()
    #将彩色的图像转换成灰度，从此可以看出read到的应该是每一帧的图像，frame是图片
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("capture",gray)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

