import cv2

# get blur value
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# the bigger value, the less blur
def blur_value(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def main():
    p1 = 'blur.png'
    p2 = 'not_blur.png'

    im1 = cv2.imread(p1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(p2, cv2.IMREAD_COLOR)

    v1 = blur_value(im1)
    v2 = blur_value(im2)


    print('blur image blur value: {0}'.format(v1))
    print('no-blur image blur value: {0}'.format(v2))
    '''
    blur image blur value: 173.16485364913112
    no-blur image blur value: 1676.2236927166962
    '''
    
    
if __name__=='__main__':
    main()