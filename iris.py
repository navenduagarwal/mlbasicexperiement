import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
test_idx = [0, 50, 100]

# training_data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)  # if axis not specified delete happens on flattened array

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# graph
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# predict
print(test_target)
print(clf.predict(test_data))
