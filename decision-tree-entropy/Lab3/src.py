import numpy as np
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

#
# Task 1
#
data = np.array(
    [
        # Study hours, Sleep hours, # Pass/Fail
        [2, 5, 1],
        [3, 6, 0],
        [5, 6, 1],
        [6, 5, 1],
        [8, 7, 1],
        [1, 4, 0],
        [4, 6, 1],
        [7, 8, 1],
        [2, 6, 1],
        [5, 5, 1],
    ]
)

X, y = data[:, :-1], data[:, -1]


#
# Task 2
#
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())


#
# Task 3
#
def best_split(X, y):
    best_feature, best_threshold, min_entropy = None, None, float("inf")

    # for every feature (study_hours/sleep_hours)
    for feature in range(X.shape[1]):

        # get all unique values for this column (feature)
        thresholds = np.unique(X[:, feature])

        # try every unique value as a split threshold
        for threshold in thresholds:

            # bitfield(s) for values above/below the threshold
            left_mask = X[:, feature] <= threshold
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
                best_feature, best_threshold, min_entropy = (
                    feature,
                    threshold,
                    weighted_entropy,
                )

    if best_feature is not None:
        return best_feature, best_threshold

    return None, None


#
# Task 4
#
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
        feature, threshold = best_split(X, y)

        # if no valid split found, return the most common
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        # Split the dataset into left & right children
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_branch = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_branch = self.fit(X[right_mask], y[right_mask], depth + 1)

        # Return the 'decision' this node will take as a dictionary
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_branch,
            "right": right_branch,
        }

    def train(self, X, y):
        self.tree = self.fit(X, y)

    #
    # Task 5
    #
    def predict_sample(self, node, X):
        # if node is a 'dict' object,
        if isinstance(node, dict):

            # go to it's left or right child based on its values
            feature, threshold = node["feature"], node["threshold"]

            if X[feature] <= threshold:
                return self.predict_sample(node["left"], X)
            else:
                return self.predict_sample(node["right"], X)

        # If node is not a dict, it must be an int, in which case we are at a leaf
        else:
            return node

    def predict(self, X):
        return np.array([self.predict_sample(self.tree, x) for x in X])


#
# Task 6
#
def draw_tree(node, graph=None, parent=None, edge_label=""):
    if graph is None:
        graph = nx.DiGraph()

    node_label = (
        f"F{node['feature']} <= {node['threshold']}"
        if isinstance(node, dict)
        else f"Class: {node}"
    )

    node_id = len(graph.nodes)

    graph.add_node(node_id, label=node_label)

    if parent is not None:
        graph.add_edge(parent, node_id, label=edge_label)
    if isinstance(node, dict):
        draw_tree(node["left"], graph, node_id, "Yes")
        draw_tree(node["right"], graph, node_id, "No")

    return graph


def plot_tree(tree):
    graph = draw_tree(tree)
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    plt.figure(figsize=(10, 6))

    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=labels,
        node_size=3000,
        node_color="lightgreen",
    )

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.savefig("decision_tree.png")


dt = DecisionTree(3)
dt.train(X, y)
predictions = dt.predict(X)
print("Predictions:", predictions)
plot_tree(dt.tree)
