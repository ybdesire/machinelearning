import cv2
import numpy as np
img_rgb = cv2.imread('tmp1.jpg', cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE, IMREAD_COLOR
img_tmp = img_rgb.copy()

lower =(0, 160, 160) # lower bound for each channel
upper = (50, 180, 180) # upper bound for each channel

# create the mask and use it to change the colors
mask = cv2.inRange(img_tmp, lower, upper)

print(np.count_nonzero(mask))# zero count is yellow color pixel number

