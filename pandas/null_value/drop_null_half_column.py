import pandas as pd

df = pd.read_csv("test.csv")
print(df)
# drop the column if the missing_value count > 0.5*full_count
half_count = len(df)/2 
df2 = df.dropna(thresh = half_count, axis = 1 ) 
print(df2)

