from sklearn.cluster import KMeans
import numpy as np


# dataset
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
print(X.shape)#(6, 2)

# cluster
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# get labels
labels = kmeans.labels_
print(labels)# [0 0 0 1 1 1]

# get cluster centers
cc = kmeans.cluster_centers_
print(cc)# [[1. 2.] [4. 2.]]

# predict
pred = kmeans.predict([[0, 0], [4, 4]])
print(pred)# [0 1]



