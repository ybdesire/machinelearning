import numpy as np
import cv2 as cv
img = cv.imread('messi.png')
mask = cv.imread('mask.png',0)
dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
