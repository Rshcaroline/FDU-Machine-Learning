from __future__ import division
import numpy as np
import os

from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import scale
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
# import pydotplus

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()

# fine_tune for KNN
parser.add_argument('--p', type=int, default=2, help="Choose p value")
parser.add_argument('--n', type=int, default=5, help="Choose n_neighbors")

# fine_tune for SVM
# parser.add_argument('--c', type=float, default=1, help="Choose penalty parameter C of the error term.")
# parser.add_argument('--k', default='rbf', help="Specifies the kernel type to be used in the algorithm.")
# parser.add_argument('--d', type=int, default=3, help="Degree of the polynomial kernel function ('poly').")

# fine_tune for DT
# parser.add_argument('--l', type=int, default=2, help="The maximum depth of the tree.")
# parser.add_argument('--n', type=int, default=5, help="Grow a tree with ``max_leaf_nodes`` in best-first fashion.")

args = parser.parse_args()

data_dir = "../anthentication/data"

X = np.load(os.path.join(data_dir, "full_features_std.npy"))
y = np.load(os.path.join(data_dir, "full_labels.npy"))

X_split = np.load(os.path.join(data_dir, "patch_features_std.npy"))
y_split = np.load(os.path.join(data_dir, "patch_labels.npy"))

total = y.shape[0]
num_T = y.sum()
num_F = total - num_T

loo = LeaveOneOut()

correct = 0
tp = 0
tn = 0
incorrect_id = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(54, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


for train_index, test_index in loo.split(X):
    X_train, X_test = X_split[train_index], X[test_index]
    X_train = np.reshape(X_train, [-1, 54])

    y_train, y_test = y_split[train_index], y[test_index]
    y_train = np.reshape(y_train, [-1])

    # ----Traditional---
    clf = KNeighborsClassifier(n_neighbors=args.n, weights="uniform", p=args.p)
    # clf = svm.SVC(C=args.c, kernel=args.k, degree=args.d)
    # clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    judgement = (clf.predict(X_test) == y_test)

    # plot the decision tree

    # Method 1
    # feature_names = [str(i) for i in range(0, 54)]
    # target_names = ["0", "1"]
    # with open("dt.dot", 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f, feature_names=feature_names,
    #                      class_names=target_names)

    # Method 2
    # dot_data = StringIO()
    # tree.export_graphviz(clf,
    #                      out_file=dot_data,
    #                      feature_names=feature_names,
    #                      class_names=target_names,
    #                      filled=True, rounded=True,
    #                      impurity=False)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("dt.pdf")

    # ----Neural Network---
    # net = Net()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # for epoch in range(1000):
    #     net.zero_grad()
    #     input = Variable(torch.Tensor(X_train))
    #     target = Variable(torch.LongTensor(y_train))
    #     outputs = net(input)
    #     loss = criterion(outputs, target)
    #     loss.backward()
    #     optimizer.step()
    # outputs = net(Variable(torch.Tensor(X_test)))
    # _, predicted = torch.max(outputs.data, 1)
    # judgement = (predicted.numpy() == y_test)

    if judgement:
        correct += 1
        if y_test:
            tp += 1
        else:
            tn += 1
    else:
        incorrect_id.append(list([test_index]))
    # print("TEST:", test_index, judgement)

output = open('test.txt', 'a')  # append model

output.write(str(args) + "\n")
output.write("Final Accuracy: %.3f [%d/%d] \n" % (correct / total, correct, total))
output.write("True Positive: %.3f [%d/%d] \n" % (tp / num_T, tp, num_T))
output.write("True Negative: %.3f [%d/%d] \n\n" % (tn / num_F, tn, num_F))

# print incorrect_id

output.close()
