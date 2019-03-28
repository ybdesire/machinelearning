import pandas as pd

df = pd.read_csv("test.csv")
print('original data')
print(df)

def coded_to_num(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

df["c5"] = coded_to_num(df["c5"], {'A':0,'B':1,'C':2})
print('modified data')
print(df)
