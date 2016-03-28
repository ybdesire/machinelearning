from numpy import *

randMat = mat(random.rand(4,4))
invRandMat = randMat.I#inverse
myResult = randMat*invRandMat

error = myResult - eye(4)
print(error)