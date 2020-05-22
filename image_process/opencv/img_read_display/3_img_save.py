import numpy as np
import cv2

# read image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)# IMREAD_GRAYSCALE, IMREAD_COLOR
print(img.shape)

# first window, display origin image
cv2.namedWindow('image-origin', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image-origin',img)

# second window, display scaled image
cv2.namedWindow('image-scale', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image-scale',img[50:150, 100:300])

print(img[50:150, 100:300])
# save image
cv2.imwrite('cut_lena.png', img[50:150, 100:300])

cv2.waitKey(0)
cv2.destroyAllWindows()

