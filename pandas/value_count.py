import pandas as pd

df = pd.read_csv("test.csv")
# count special column values distribution
print(pd.value_counts(df['c3']))







