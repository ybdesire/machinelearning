import numpy as np
import cv2


def _flatten_rgb(img):
    # img is MxNx3, then r/g/b is MxN
    r, g, b = cv2.split(img)
    
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))

    r[y_filter], g[y_filter] = 255, 255
    b[np.invert(y_filter)] = 0

    b[b_filter], b[np.invert(b_filter)] = 255, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 255, 0

    flattened = cv2.merge((r, g, b))
    return flattened

    

# read image
img = cv2.imread('non_flattern.jpg', cv2.IMREAD_COLOR)# IMREAD_GRAYSCALE, IMREAD_COLOR
# flattern
img_flattern = _flatten_rgb(img)
# save image
cv2.imwrite('flattern.jpg', img_flattern)
