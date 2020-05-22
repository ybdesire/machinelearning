import numpy as np
import cv2


img = cv2.imread('connectedComponents.png', cv2.IMREAD_GRAYSCALE)# IMREAD_GRAYSCALE, IMREAD_COLOR
labels,result = cv2.connectedComponents(img)
print(labels)
# 5 component
# 4 white, 1 black
# label int in result
