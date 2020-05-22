import cv2

img_rgb = cv2.imread('tmp.jpg', cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE, IMREAD_COLOR
img_tmp = img_rgb.copy()

for i in range(img_tmp.shape[0]):
    for j in range(img_tmp.shape[1]):
        x0 = img_tmp[i][j][0]
        x1 = img_tmp[i][j][1]
        x2 = img_tmp[i][j][2]
        # replace yellow color to black
        if(x0<=50):
            if(x1>=160 and x1<=180):
                if(x2>=160 and x1<=180):
                    img_tmp[i][j][0] = 0
                    img_tmp[i][j][1] = 0
                    img_tmp[i][j][2] = 0

cv2.imshow('img_tmp', img_tmp)
cv2.waitKey(0)