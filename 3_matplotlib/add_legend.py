import math
import matplotlib.pyplot as plt
import numpy as np

# Generate a sinusoid
nbSamples = 256

x = np.linspace(-math.pi, math.pi, num=nbSamples)
y = np.sin(x)
z = np.cos(x)
    
plt.plot(x,y,label='sin(x)')
plt.plot(x,z,label='cos(x)')
plt.legend(loc='best')
plt.show()
