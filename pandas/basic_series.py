import pandas as pd

# Series is 1D data at pd
series1 = pd.Series([2,4,6,8,10])
print(series1)
print("series1.values: {0}\n".format(series1.values))
print("series1.index: {0}\n".format(series1.index))

# add index for Series
series2 = pd.Series([2,4,6,8,10], index=["C", "D", "E", "F", "G"])
print("series2:\n{}\n".format(series2))
print("E is {}\n".format(series2["E"]))

