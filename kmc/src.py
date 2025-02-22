from math import sqrt
from random import uniform
import copy

def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def k_means_clustering(points, k):
    # Get min, max points
    min_x = min(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_x = max(points, key=lambda x: x[0])[0]
    max_y = max(points, key=lambda x: x[1])[1]

    # Generate `k` random centroids b/w min & max
    centroids = [(uniform(min_x, max_x), uniform(min_y, max_y)) for _ in range(k)]

    clusters = [[] for _ in range(k)]
    backup_mapping = None

    while True:
        backup_mapping = copy.deepcopy(clusters)
        clusters = [[] for _ in range(k)]

        # Place every point in the list of it's nearest centroid
        for p in points:
            closest_centroid_idx = min(range(k), key=lambda i: euclidean_distance(p, centroids[i]))
            clusters[closest_centroid_idx].append(p)

        # If no change, we're done
        if clusters == backup_mapping:
            break

        # Update centroids
        for i in range(k):
            if clusters[i]:
                sum_x = sum(point[0] for point in clusters[i])
                sum_y = sum(point[1] for point in clusters[i])
                centroids[i] = (sum_x / len(clusters[i]), sum_y / len(clusters[i]))

    return centroids

points = [
    (2, 3), (3, 4), (4, 5), (10, 10), (11, 11), (12, 12),
    (50, 50), (51, 51), (52, 52), (20, 25), (22, 24), (23, 26)
]

print(k_means_clustering(points, 3))
