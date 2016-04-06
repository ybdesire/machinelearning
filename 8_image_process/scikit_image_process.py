import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.ndimage as ndimage


def bytes_from_file(filename, chunksize=800):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break
    
def get_one_imagefrom_mnist():
    filename = "mnist-one-image";
    count = 0;
    image = []
    for b in bytes_from_file(filename):
        if(count<16):
            pass
        else:
            image.append(b)
        count+=1
    return image

          
def smooth():
    image_list = get_one_imagefrom_mnist()
    image_array =np.asarray(image_list)
    image =image_array.reshape(28, 28)
    
    plt.imshow(image, cmap=cm.binary)
    plt.show()
    
    for i in range(11):
        image = ndimage.gaussian_filter(image, sigma=float(i/10))
        print(float(i/10))
        plt.imshow(image, cmap=cm.binary)
        plt.show()

def closing():
    image_list = get_one_imagefrom_mnist()
    image_array =np.asarray(image_list)
    image =image_array.reshape(28, 28)
    
    ndimage.binary_closing(image, structure=np.ones((2,2))).astype(int)
    plt.imshow(image, cmap=cm.binary)
    plt.show()

def dilation():
    image_list = get_one_imagefrom_mnist()
    image_array =np.asarray(image_list)
    image =image_array.reshape(28, 28)
    
    ndimage.binary_dilation(image).astype(int)
    plt.imshow(image, cmap=cm.binary)
    plt.show()

def erosion():
    image_list = get_one_imagefrom_mnist()
    image_array =np.asarray(image_list)
    image =image_array.reshape(28, 28)
    
    ndimage.binary_erosion(image).astype(int)
    plt.imshow(image, cmap=cm.binary)
    plt.show()
    
    
def main():
    image_list = get_one_imagefrom_mnist()
    image_array =np.asarray(image_list)
    image =image_array.reshape(28, 28)
    
   
    plt.subplot(1, 2, 1) 
    plt.title('original')
    plt.imshow(image, cmap=cm.binary)
        
        
    image = ndimage.shift(image, (2,3)).astype(int)
	
    plt.subplot(1, 2, 2)    
    plt.title('shift')
    plt.imshow(image, cmap=cm.binary)
	
    plt.show()

if __name__ == "__main__":
    main()