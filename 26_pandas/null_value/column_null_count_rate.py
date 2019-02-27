import pandas as pd

df = pd.read_csv("test.csv")

check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df)) #null_data/missing_value_data rate
print(check_null[check_null > 0.2]) # check the data with null data > 20%


