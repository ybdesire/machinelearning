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
print(a[2][1])
# result is [2 2 2 3] [1 3 5 0], that is
# a[2][1] = 2
# a[2][3] = 2
# a[2][5] = 2
# a[3][0] = 2
