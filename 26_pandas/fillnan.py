import pandas as pd
import numpy as np

df = pd.DataFrame([[1.0, np.nan, 3.0, 4.0],
                  [5.0, np.nan, np.nan, 8.0],
                  [9.0, np.nan, np.nan, 12.0],
                  [13.0, np.nan, 15.0, 16.0]])
                  
print(df)

df2 = df.fillna(100)# replace all nan to 100

print(df2)

                  