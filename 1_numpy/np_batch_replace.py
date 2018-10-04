import numpy as np


a = np.array(
    [
        [1,0,1,0,1,0,1],
        [0,0,0,1,1,1,1],
        [1,2,1,2,1,2,1],
        [2,3,3,3,3,3,3]
    ]
)

y,x = np.where(a==2)#find the index of  2 in a
print(y,x)
# result is [2 2 2 3] [1 3 5 0], that is
# a[2][1] = 2
# a[2][3] = 2
# a[2][5] = 2
# a[3][0] = 2

a[y,x]=555# all 2 in a where be replaced to 555
print(a)