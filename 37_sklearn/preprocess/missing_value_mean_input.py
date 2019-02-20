import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

# read data
df = pd.read_csv("test.csv")
print('original data')
print(df)
numColumns = df.select_dtypes(include=[np.number]).columns
print('numColumns ', numColumns)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df[numColumns])
df[numColumns] = imr.transform(df[numColumns])

print('final data')
print(df)



