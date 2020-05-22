from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# feature selection by chi2
# more details about chi2: http://www.cnblogs.com/emanlee/archive/2008/10/25/1319569.html
x_data,y_data = load_digits(return_X_y=True)
print(x_data.shape)#(1797, 64)
# selection 20 features from x_data
x_new = SelectKBest(chi2, k=20).fit_transform(x_data,y_data)
print(x_new.shape)#(1797, 20)