import pandas as pd

df = pd.DataFrame([[13,0,0,0], [0,0,18,0]])
print(df)
'''
    0  1   2  3
0  13  0   0  0
1   0  0  18  0
'''

# "delete" the zero-columns
df2 = df.loc[:, (df != 0).any(axis=0)]
print(df2)
'''
    0   2
0  13   0
1   0  18
'''



