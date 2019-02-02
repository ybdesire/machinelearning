import pandas as pd


index = [0,1,2]
columns = ['c1','c2','c3','c4']

data = [
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
]
df = pd.DataFrame(data, index=index, columns=columns)

print(df)
'''
   c1  c2  c3  c4
0   1   1   1   1
1   2   2   2   2
2   3   3   3   3
'''
