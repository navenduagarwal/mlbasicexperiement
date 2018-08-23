from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, point):
        best_dist = euc(point, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            distance = euc(point, self.X_train[i])
            if distance < best_dist:
                best_dist = distance
                best_index = i
        return self.y_train[best_index]

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

# 1 decision tree
my_classifier = tree.DecisionTreeClassifier()

# 2 k neighbours classifier
my_classifier_2 = KNeighborsClassifier()

# 3 custom scrappy KNN
my_classifier_3 = ScrappyKNN()

train(my_classifier, X_train, y_train)
train(my_classifier_2, X_train, y_train)
train(my_classifier_3, X_train, y_train)

predictions, accuracy = predict(my_classifier, X_test, y_test)
predictions_2, accuracy_2 = predict(my_classifier_2, X_test, y_test)
predictions_3, accuracy_3 = predict(my_classifier_3, X_test, y_test)

print("real labels :%s" % (y_test))
print("first predictions: %s, accuracy: %s" % (predictions, accuracy))
print("second predictions: %s, accuracy: %s" % (predictions_2, accuracy_2))
print("third predictions: %s, accuracy: %s" % (predictions_3, accuracy_3))
