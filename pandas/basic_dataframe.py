import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.arange(16).reshape(4,4),
    columns=["column1", "column2", "column3", "column4"],
    index=["a", "b", "c", "d"])
print("df2:\n{}\n".format(df2))
print("df2[column2][d]={0}".format(df2["column2"]["d"]))