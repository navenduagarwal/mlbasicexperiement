# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

# data split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# train
def train(classifier, features, labels):
    return classifier.fit(features, labels)


# predictions & accuracy score
from sklearn.metrics import accuracy_score


def predict(classifier, features, y_true):
    predictions = classifier.predict(features)
    accuracy = accuracy_score(y_true, predictions)
    return predictions, accuracy


# classifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# 1 type
my_classifier = tree.DecisionTreeClassifier()
my_classifier_2 = KNeighborsClassifier()

# 2 type
train(my_classifier, X_train, y_train)
train(my_classifier_2, X_train, y_train)

predictions, accuracy = predict(my_classifier, X_test, y_test)
predictions_2, accuracy_2 = predict(my_classifier_2, X_test, y_test)
print(y_test)
print(predictions)
print(accuracy)
print(predictions_2)
print(accuracy_2)
