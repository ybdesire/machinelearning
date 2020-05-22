import cv2

img_path_0 = 'eagle_2018_09_05_18_40_13_657.jpg'
img_path_1 = 'eagle_2018_09_05_18_40_13_723.jpg'
img_path_2 = 'eagle_2018_09_05_18_40_13_790.jpg'

img_rgb_0 = cv2.imread(img_path_0, cv2.IMREAD_COLOR)
img_rgb_1 = cv2.imread(img_path_1, cv2.IMREAD_COLOR)
img_rgb_2 = cv2.imread(img_path_2, cv2.IMREAD_COLOR)

img_hsv_0 = cv2.cvtColor(img_rgb_0, cv2.COLOR_RGB2HSV)
img_hsv_1 = cv2.cvtColor(img_rgb_1, cv2.COLOR_RGB2HSV)
img_hsv_2 = cv2.cvtColor(img_rgb_2, cv2.COLOR_RGB2HSV)

cv2.imshow('img_hsv_0', img_hsv_0)
cv2.imshow('img_hsv_0[0]', img_hsv_0[:,:,0])
cv2.imshow('img_hsv_0[1]', img_hsv_0[:,:,1])
cv2.imshow('img_hsv_0[2]', img_hsv_0[:,:,2])

cv2.waitKey(0)

