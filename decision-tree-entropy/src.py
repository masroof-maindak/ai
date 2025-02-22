import numpy as np
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

#
# Task 1
#
data = np.array([
    [2, 5, 1], [3, 6, 0], [5, 6, 1], [6, 5, 1], [8, 7, 1], [1, 4, 0], [4, 6, 1], [7, 8, 1], [2, 6, 1], [5, 5, 1]
])

x, y = data[:, :-1], data[:, -1]

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
def best_split(x, y):
    best_feature, best_threshold, min_entropy = None, None, float('inf')

    for feature in range(x.shape[1]):
        thresholds = np.unique(x[:, feature])

        for threshold in thresholds:
            left_mask = x[:, feature] <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_entropy = entropy(y[left_mask])
            right_entropy = entropy(y[right_mask])

            weighted_entropy = (left_mask.sum() * left_entropy + right_mask.sum() * right_entropy) / len(y)

            if weighted_entropy < min_entropy:
                best_feature, best_threshold, min_entropy = feature, threshold, weighted_entropy

    return best_feature, best_threshold


#
# Task 4
#
class DecisionTree:
    def __init__(self, tree_depth):
        self.depth = tree_depth
        self.tree = None

    def fit(self, x, y, depth=0):
        if len(set(y)) == 1:
            return y[0]
        if self.depth is not None and depth >= self.depth:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = best_split(x, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask

        left_branch = self.fit(x[left_mask], y[left_mask], depth + 1)
        right_branch = self.fit(x[right_mask], y[right_mask], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_branch, 'right': right_branch}

    def train(self, x, y):
        self.tree = self.fit(x, y)
    #
    # Task 5
    #
    def predict_sample(self, node, x):
        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self.predict_sample(node['left'], x)
            else:
                return self.predict_sample(node['right'], x)
        else:
            return node

    def predict(self, x):
        return np.array([self.predict_sample(self.tree, x) for x in x])
#
# Task 6
#
def draw_tree(node, graph=None, parent=None, edge_label=''):
    if graph is None:
        graph = nx.DiGraph()
    node_label = f"F{node['feature']} <= {node['threshold']}" if isinstance(node, dict) else f"Class: {node}"
    node_id = len(graph.nodes)
    graph.add_node(node_id, label=node_label)
    if parent is not None:
        graph.add_edge(parent, node_id, label=edge_label)
    if isinstance(node, dict):
        draw_tree(node['left'], graph, node_id, 'Yes')
        draw_tree(node['right'], graph, node_id, 'No')
    return graph

def plot_tree(tree):
    graph = draw_tree(tree)
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightgreen')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.savefig("decision_tree.png")

dt = DecisionTree(3)
dt.train(x, y)
predictions = dt.predict(x)
print("Predictions:", predictions)
plot_tree(dt.tree)

