import numpy as np
from scipy.spatial.distance import cdist


x1 = np.array([[1,1]])
x2 = np.array([[4,5]])
distance = cdist(x1,x2,"euclidean")
print(distance)#[[ 5.]]
