from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

# feature selection by ref
x_data,y_data = load_iris(return_X_y=True)
print(x_data.shape)#(150, 4)
# selection 2 features from x_data
estimator = SVR(kernel="linear")
selector = RFE(estimator, 2, step=1)
selector = selector.fit( x_data,y_data )
print( selector.support_  )# [False False  True  True]
print( selector.ranking_ )# [2 3 1 1]
