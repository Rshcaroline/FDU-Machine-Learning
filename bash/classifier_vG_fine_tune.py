from __future__ import division
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

import argparse

parser = argparse.ArgumentParser()

# fine_tune for KNN
# parser.add_argument('--p', type=int, default=2, help="Choose p value")
# parser.add_argument('--n', type=int, default=5, help="Choose n_neighbors")

# fine_tune for SVM
# parser.add_argument('--c', type=float, default=1, help="Choose penalty parameter C of the error term.")
# parser.add_argument('--k', default='rbf', help="Specifies the kernel type to be used in the algorithm.")
# parser.add_argument('--d', type=int, default=3, help="Degree of the polynomial kernel function ('poly').")

# fine_tune for DT
parser.add_argument('--l', type=int, default=2, help="The maximum depth of the tree.")
parser.add_argument('--n', type=int, default=5, help="Grow a tree with ``max_leaf_nodes`` in best-first fashion.")

args = parser.parse_args()


# features_dir = "../anthentication/data/full_features_norm_std.npy"
features_dir = "../neural-transfer-master/style_features_256.npy"
labels_dir = "../anthentication/data/full_labels.npy"
X = np.load(features_dir)
y = np.load(labels_dir)

# full_features_std
# INDEX = [32, 33, 34, 24, 28, 30]

# full_features_norm_std
# INDEX = [0, 2, 36, 39, 49, 18, 19, 20, 21]
# X = X[:, INDEX]

loo = LeaveOneOut()

correct = 0
tp = 0
tn = 0
incorrect_id = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # clf = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1)
    # clf = KNeighborsClassifier(n_neighbors=args.n, weights="uniform", p=args.p)
    # clf = svm.SVC(C=args.c, kernel=args.k, degree=args.d)
    # clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=2)
    clf = DecisionTreeClassifier(max_depth=args.l, max_leaf_nodes=args.n)
    clf.fit(X_train, y_train)
    judgement = clf.predict(X_test) == y_test
    # print(int(clf.predict(X_test)[0]), int(y_test[0]))
    if judgement:
        correct += 1
        if y_test:
            tp += 1
        else:
            tn += 1
    else:
        incorrect_id.append(list([test_index]))
    # print("TEST:", test_index, judgement)

output = open('dt.txt', 'a')  # append model

output.write(str(args) + "\n")
output.write("Final Accuracy: %.3f [%d/21] \n" % (correct / 21, correct))
output.write("True Positive: %.3f [%d/12] \n" % (tp / 12, tp))
output.write("True Negative: %.3f [%d/9] \n\n" % (tn / 9, tn))

# print incorrect_id

output.close()