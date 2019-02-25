import pandas as pd

df = pd.read_csv("test.csv")
print(df)
# drop the column if it contains one missing_value
df2 = df.dropna(axis=1)
print(df2)

