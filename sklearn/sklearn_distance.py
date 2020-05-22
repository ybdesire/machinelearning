from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import paired_euclidean_distances


x = [[1,2,3,4,5], [1,2,3,4,4], [1,2,3,4,3]]
y = [[1,2,3,4,5]]
dist1 = euclidean_distances(x, y)
#dist2 = paired_euclidean_distances(x, y)


print(dist1)
#print(dist2)