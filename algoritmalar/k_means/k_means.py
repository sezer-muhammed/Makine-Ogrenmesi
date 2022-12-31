import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#create x, y points that are in clusters

def create_clustered_data(N, k):
    points_per_cluster = float(N)/k
    X = []
    for i in range(k):
        income_centroid = np.random.uniform(20000.0, 200000.0)
        age_centroid = np.random.uniform(20.0, 70.0)
        for j in range(int(points_per_cluster)):
            X.append([np.random.normal(income_centroid, 10000.0), np.random.normal(age_centroid, 2.0)])
    X = np.array(X)
    return X

x, y = create_clustered_data(100, 3).T



plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

    plt.scatter(x, y, c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red')
    plt.title('K-means with {} clusters'.format(i))
    plt.show()

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()