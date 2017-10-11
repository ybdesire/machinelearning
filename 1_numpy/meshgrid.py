import numpy as np

x=[1,2,3]
y=[4,5,6]
xx, yy = np.meshgrid(x,y)

print(xx)
'''
[[1 2 3]
 [1 2 3]
 [1 2 3]]
'''


print(yy)
'''
[[4 4 4]
 [5 5 5]
 [6 6 6]]
'''