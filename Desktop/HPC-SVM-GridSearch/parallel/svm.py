import numpy as np
from sklearn.svm import SVC

class SVM:
    def __init__(self, C=1.0, gamma='scale', kernel='rbf'):
        self.model = SVC(C=C, gamma=gamma, kernel=kernel)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == y_test)
