import numpy as np
import cv2


# read image (flatterned image is better)
img = cv2.imread('flattern.jpg', cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE, IMREAD_COLOR

# img is MxNx3, then r/g/b is MxN
r, g, b = cv2.split(img)


cv2.namedWindow('r', cv2.WINDOW_AUTOSIZE)
cv2.imshow('r',r)
cv2.namedWindow('g', cv2.WINDOW_AUTOSIZE)
cv2.imshow('g',g)
cv2.namedWindow('b', cv2.WINDOW_AUTOSIZE)
cv2.imshow('b',b)

cv2.waitKey(0)#esc
cv2.destroyAllWindows()
