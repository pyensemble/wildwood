


import pandas as pd
from sklearn.datasets import make_circles

from wildwood._utils import NP_DOUBLE_t

from wildwood._classes import DecisionTreeClassifier

n_samples = 1500
random_state = 42


X, y = make_circles(n_samples=n_samples, noise=0.2, random_state=random_state)
y = y.astype(NP_DOUBLE_t)

clf = DecisionTreeClassifier()

print(clf)

clf.fit(X, y)

from wildwood._tree import print_tree, get_nodes

print_tree(clf.tree_)


nodes = get_nodes(clf.tree_)

print(nodes)

# # print(y[:10])
#
# from wildwood._tree import Tree, print_tree, get_nodes, tree_add_node
#
# from wildwood._classes import DecisionTreeClassifier
#
#
# from wildwood._utils import Stack
#
# from wildwood._utils import SIZE_MAX


#
# @njit
# def main():
#     n_features = 12
#     n_classes = np.array([3], dtype=NP_SIZE_t)
#     n_outputs = 1
#
#     tree = Tree(n_features, n_classes, n_outputs)
#
#     parent = 0
#     is_left = 1
#     is_leaf = 1
#     feature = 11
#     threshold = 3.14
#     impurity = 2.78
#     n_node_samples = 42
#     weighted_n_node_samples = 2.32
#
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#
#     return tree
#
# tree = main()
#
# pd.set_option("display.max_columns", 20)
# pd.set_option("display.width", 150)
#
# print_tree(tree)
#
# # print(tree.nodes)
#
# print(get_nodes(tree))

# main()

# print(SIZE_MAX.max)
