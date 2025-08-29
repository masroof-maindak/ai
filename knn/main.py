from math import sqrt
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
)  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from heapq import (
    nsmallest,
)  # https://docs.python.org/3/library/heapq.html#heapq.nsmallest


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix  # , f1_score

X, y = load_iris(return_X_y=True)


# scaler = StandardScaler()
# # CHECK: when to use this or MinMaxScaler?
# X = scaler.fit_transform(X, y)


# Stratify ensures that the ratio b/w categories/targets remains the same as in the original state
# Significant when there's a lot of class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# CHECK: how significant is stratification? i.e how skewed is this dataset anyway?


knn = KNeighborsClassifier(n_neighbors=3)
# CHECK: try different norm/p and how it impacts training
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

"""
Minkowski Distance -> `underroot-'p'( (x1 - y1)^p + (x2 - y2)^p + ... + (xn - yn)^p)`

When p == 2, it's Euclidean distance; p == 1? Manhattan distance.

p i.e the norm, is the power set to the root and 'internal squares' in the formula.

Q: What happens at p = inf?
>> Observe that as we jump from p = 1 to p = 2, the magnitude of an
   'internal distance' in one direction balloons. At infinity, this effect
   is exaggerated even further and only the single greatest internal square
   remains, with the others zeroing out.

Q: What does a circle look like in Manhattan distance
>> Rhombus i.e 45 degree square since every point has a distance of 1 from the center
>> CHECK: how?

Q: What does a circle look like at p = inf?
>> Square. How?
>> Problem: I was looking at a circle as a literal circle
>> So let's first identify what *defines* a circle?
>> Set of points at a distance of 1 from the 0,0 point
>> Thus, for all the cardinal directions, we get `underroot-inf((1-0)^inf + (0-0)^inf)`
>> This is impossible to process but with some calculus magic, we derive that this is equal to one
>> For every possible point on the square, say 1 and 0.6, as we exponentiate each to infinity
>> the 0.6 would get zeroed out, and only the max is propagated, i.e 1**inf, which is 1. Sex.

Retrospective: I merely *understood* how the aforementioned works.
Q: All I did How could *I* have derived it on my own,
   without the answer being handed on a silver platter?
   TODO: Discuss w/ Mudassir.
>> I forgot what he said, but the core idea was that if we span out vectors from the origin of a
   circle towards its infinitude of sides, we will observe that the vectors preserve their direction
   across different values of `p` -- the same can not be said for their magnitude.
"""

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# ----------------------
# Manual Implementation
# ----------------------


def euclidean_distance_4d(p1, p2):
    return sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + (p1[2] - p2[2]) ** 2
        + (p1[3] - p2[3]) ** 2
    )


def get_k_neighbors(x_train, y_train, test_sample, k):
    if len(x_train) != len(y_train):
        raise ValueError("Number of features & labels is unequal!")

    # Calculate distance from test sample to all training samples
    # Time Complexity: n * k * d
    dists = []
    for i in range(len(x_train)):
        dists.append((euclidean_distance_4d(x_train[i], test_sample), y_train[i]))

    # Sort first `k` based on distance
    # Time Complexity: n
    partially_sorted = nsmallest(k, dists, key=lambda dist: dist[0])
    return partially_sorted


"""
As we spread out into higher and higher dimensions, the average distance
b/w points is going to increase. E.g on a line from 0 to 1, the max distance is 1.

In a square, the max distance is 1.41 (sqrt(2)).

This can be extrapolated to any number of dimensions.

HOWEVER, the DEVIATION from the average distance goes down.

---

This may lead to KNN being dogshit when the dimensionality is high. One point in
particular may be uselessly far away from everyone else, and thus pulls a lot
of other points in.
"""


def predict_classification(x_train, y_train, test_sample, k):
    # Generally, a more appropriate data structure used is a 'kd-tree'
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
for i in range(len(X_test)):
    if predict_classification(X_train, y_train, X_test[i], 5) == y_test[i]:
        correct += 1

print(correct / len(X_test))
