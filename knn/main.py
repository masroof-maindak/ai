from math import sqrt
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from heapq import nsmallest # https://docs.python.org/3/library/heapq.html#heapq.nsmallest

iris = load_iris()
x = iris.data   # Features
y = iris.target # Labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def euclidean_distance_4d(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)

def get_k_neighbors(x_train, y_train, test_sample, k):
    if (len(x_train) != len(y_train)):
        raise ValueError('Number of features & labels is unequal!')

    # Calculate distance from test sample to all training samples
    dists = []
    for i in range(len(x_train)):
        dists.append(( euclidean_distance_4d( x_train[i], test_sample), y_train[i] ))

    # Sort first `k` based on distance
    partially_sorted = nsmallest(k, dists, key=lambda dist: dist[0])
    return partially_sorted

def predict_classification(x_train, y_train, test_sample, k):
    knns = get_k_neighbors(x_train, y_train, test_sample, k)

    counter = defaultdict(int)
    for neighbour in knns:
        counter[neighbour[1]] += 1

    a = max(counter.items(), key=lambda a: a[1])
    return a[0]

#
# Testing
#
correct = 0
for i in range(len(x_test)):
    if predict_classification(x_train, y_train, x_test[i], 5) == y_test[i]:
        correct += 1

print(correct / len(x_test))

