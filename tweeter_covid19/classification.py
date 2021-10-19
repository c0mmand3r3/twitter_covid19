"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/15/2019
"""

import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class Classification:
    def __init__(self, activate_svm=True):
        self.svm_model = None
        if activate_svm is True:
            self.svm_model = svm.SVC()

    def fit_parameter(self, c, kernel, gamma):
        """
        :param c:
        :param kernel:
        :param gamma:
        :return:
        """
        self.svm_model.C = c
        self.svm_model.kernel = kernel
        self.svm_model.gamma = gamma

    def classify(self, train_x=None, train_y=None, test_x=None, test_y=None, normalize=False):
        """
        :param normalize:
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :return:
        """
        if train_x is None or train_y is None or test_x is None or test_y is None:
            return None
        train_x = np.asarray(train_x)
        test_x = np.asarray(test_x)
        if normalize is True:
            scaler = StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)

        self.svm_model.fit(train_x, train_y)

        return self.svm_model.score(test_x, test_y)

    def fit_predict(self, train_x=None, train_y=None, test_x=None, test_y=None, normalize=False):
        """
        :param normalize:
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :return:
        """
        if train_x is None or train_y is None or test_x is None or test_y is None:
            return None
        train_x = np.asarray(train_x)
        test_x = np.asarray(test_x)
        if normalize is True:
            scaler = StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)

        self.svm_model.fit(train_x, train_y)

        return self.svm_model.predict(test_x)

    def get_confusion_matrix(self, train_x=None, train_y=None, test_x=None, test_y=None, normalize=False):
        if train_x is None or train_y is None or test_x is None or test_y is None:
            return None
        train_x = np.asarray(train_x)
        test_x = np.asarray(test_x)
        if normalize is True:
            scaler = StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)

        self.svm_model.fit(train_x, train_y)
        prediction = self.svm_model.predict(test_x)
        _confusion_matrix = confusion_matrix(test_y, prediction)
        return prediction, _confusion_matrix
