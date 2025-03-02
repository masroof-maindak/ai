import numpy as np
from collections import Counter
from enum import Enum


class Age(Enum):
    YOUNG = 0
    MIDDLE_AGED = 1
    SENIOR = 2


class Income(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class PriorPurchase(Enum):
    NO = 0
    YES = 1


"""
Patterns observed in dataset:
    1. All succesful purchases had high income and a prior purchase
    2. Age seemingly does not matter
"""
y = np.array([1, 0, 1, 0, 1])
X = np.array(
    [
        [Age.YOUNG.value, Income.HIGH.value, PriorPurchase.YES.value],
        [Age.MIDDLE_AGED.value, Income.MEDIUM.value, PriorPurchase.NO.value],
        [Age.SENIOR.value, Income.HIGH.value, PriorPurchase.YES.value],
        [Age.YOUNG.value, Income.MEDIUM.value, PriorPurchase.NO.value],
        [Age.MIDDLE_AGED.value, Income.HIGH.value, PriorPurchase.YES.value],
    ]
)


def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())


def best_split(X, y):
    best_feature, best_category, min_entropy = None, None, float("inf")

    # for every feature (age/income/prior-purchase)
    for feature in range(X.shape[1]):

        # get all unique values (categories) for this column (feature)
        categories = np.unique(X[:, feature])

        # try every unique value as a 'split threshold'
        for category in categories:

            # bitfield(s) for values that match/don't match this category
            left_mask = X[:, feature] == category
            right_mask = ~left_mask

            # if a split results in an empty set, it's pointless,
            # because if all values fall to one side, entropy is
            # not reduced.
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Calculate weighted entropy
            left_entropy = entropy(y[left_mask])
            right_entropy = entropy(y[right_mask])

            left_weight = left_mask.sum() / len(y)
            right_weight = right_mask.sum() / len(y)

            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy

            # Update best split if it reduced entropy
            if weighted_entropy < min_entropy:
                best_feature, best_category, min_entropy = (
                    feature,
                    category,
                    weighted_entropy,
                )

    if best_feature is not None:
        return best_feature, best_category

    return None, None


class DecisionTree:
    def __init__(self, tree_depth):
        self.depth = tree_depth
        self.tree = None

    # returns either a dict (internal node) or a label (leaf)
    def fit(self, X, y, depth=0):
        # if all labels are the same, we're done
        if len(set(y)) == 1:
            return y[0]

        # if max depth has been reached, return the most common label
        if self.depth is not None and depth >= self.depth:
            return Counter(y).most_common(1)[0][0]

        # Split
        feature, category = best_split(X, y)

        # if no valid split found, return the most common
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        # Split the dataset into left & right children
        left_mask = X[:, feature] == category
        right_mask = ~left_mask

        left_branch = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_branch = self.fit(X[right_mask], y[right_mask], depth + 1)

        # Return the 'decision' this node will take as a dictionary
        return {
            "feature": feature,
            "category": category,
            "left": left_branch,
            "right": right_branch,
        }

    def train(self, X, y):
        self.tree = self.fit(X, y)

    def predict_sample(self, node, X):
        # if node is a 'dict' object,
        if isinstance(node, dict):

            # go to it's left or right child based on whether or not
            # it matches the category of that node's representative feature
            feature, category = node["feature"], node["category"]

            if X[feature] == category:
                return self.predict_sample(node["left"], X)
            else:
                return self.predict_sample(node["right"], X)

        # If node is not a dict, it must be an int, in which case we are at a leaf
        else:
            return node

    def predict(self, x):
        return self.predict_sample(self.tree, x)


# Create and train decision tree
clf = DecisionTree(3)
clf.train(X, y)

# Predict all possible combinations of inputs
print(f"{'Age':<12} {'Income':<9} {'Prior Purchase':<15} {'Prediction'}")
print("=" * 49)

for age in Age:
    for income in Income:
        for prior_purchase in PriorPurchase:
            features = [age.value, income.value, prior_purchase.value]
            prediction = clf.predict(features)
            print(
                f"{age.name:<12} {income.name:<9} {prior_purchase.name:<15} {prediction}"
            )
