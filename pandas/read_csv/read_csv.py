import pandas as pd
df = pd.read_csv('data.csv')

print(df['x1'])# select column


data_x = []
data_y = []

# iterator df row by row
for index, row in df.iterrows():
    fea = [int(row['x1']), int(row['x2']), int(row['x3']) ]
    y = int(row['y'])
    data_x.append(fea)
    data_y.append(y)
    
print(data_x)