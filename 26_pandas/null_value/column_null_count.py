import pandas as pd

df = pd.read_csv("test.csv")

# null count is blank/missing_value count
def null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)
column_null_count = df.apply(null_count)
print (column_null_count)

'''
c1    4
c2    2
c3    1
c4    0
dtype: int64

