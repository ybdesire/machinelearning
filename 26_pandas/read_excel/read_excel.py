import pandas as pd

xl = pd.ExcelFile("test.xlsx")
print('sheet_names: {0}'.format(xl.sheet_names))
df = xl.parse("details")

for index, row in df.iterrows():
    name = row['name']
    age = row['age']
    country = row['country']
    print('{0},{1}'.format(country, name))
