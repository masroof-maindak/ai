import pandas as pd
from math import sqrt
from random import uniform
import copy
import matplotlib.pyplot as plt

data = pd.read_csv("./winequality-red.csv", sep=";")
X = data[["alcohol", "volatile acidity"]].values


def euclidean_dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# returns ( list of (float, float), list )
def k_means_clustering(points, k):
    # Get min, max points
    min_x = min(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_x = max(points, key=lambda x: x[0])[0]
    max_y = max(points, key=lambda x: x[1])[1]

    # Generate `k` random centroids b/w min & max
    centroids = [(uniform(min_x, max_x), uniform(min_y, max_y)) for _ in range(k)]

    clusters = [[] for _ in range(k)]

    while True:
        previous_centroids = copy.deepcopy(centroids)
        clusters = [[] for _ in range(k)]

        # Place every point in the list of it's nearest centroid
        for p in points:
            nearest_centroid_idx = min(
                range(k), key=lambda i: euclidean_dist(p, centroids[i])
            )
            clusters[nearest_centroid_idx].append(p)

        # Update centroids
        for i in range(k):
            if clusters[i]:
                sum_x = sum(point[0] for point in clusters[i])
                sum_y = sum(point[1] for point in clusters[i])
                centroids[i] = (sum_x / len(clusters[i]), sum_y / len(clusters[i]))

        # If the centroids have not deviated, we're done
        if (
            all(euclidean_dist(previous_centroids[i], centroids[i]) for i in range(k))
            <= 1e-6
        ):
            break

    return centroids, clusters


centroids, clusters = k_means_clustering(X, 3)

# Plot the clusters
for i, cluster in enumerate(clusters):
    cluster_x = [point[0] for point in cluster]
    cluster_y = [point[1] for point in cluster]
    plt.scatter(cluster_x, cluster_y, label=f"Cluster {i+1}")

# Plot the centroids
centroids_x = [centroid[0] for centroid in centroids]
centroids_y = [centroid[1] for centroid in centroids]
plt.scatter(centroids_x, centroids_y, marker="X", c="red", label="Centroids")

plt.xlabel("Alcohol")
plt.ylabel("Volatile Acidity")
plt.title("K-Means Clustering of Wine Quality Data")
plt.legend()
plt.savefig("plot.png")
