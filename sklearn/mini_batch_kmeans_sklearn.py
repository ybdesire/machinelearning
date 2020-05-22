from sklearn.cluster import MiniBatchKMeans
import numpy as np
# for large scale 
# For large scale learning (say n_samples > 10k) MiniBatchKMeans is probably much faster than the default batch implementation.


# dataset
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 0], [4, 4],
               [4, 5], [0, 1], [2, 2],
               [3, 2], [5, 5], [1, -1]])
print(X.shape)# (12, 2)

# ------usage-1------
# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=6)
kmeans = kmeans.partial_fit(X[0:6,:])
kmeans = kmeans.partial_fit(X[6:12,:])

# get cluster center
cc = kmeans.cluster_centers_
print(cc)# [[1 1]  [3 4]]

# predict
pred = kmeans.predict([[0, 0], [4, 4]])
print(pred)# [0 1]


# ------usage-2------
# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=6,max_iter=10).fit(X)

# cluster center
cc = kmeans.cluster_centers_
print(cc)# [[3.95918367 2.40816327]  [1.12195122 1.3902439 ]]

# predict
pred = kmeans.predict([[0, 0], [4, 4]])
print(pred)# [1 0]

