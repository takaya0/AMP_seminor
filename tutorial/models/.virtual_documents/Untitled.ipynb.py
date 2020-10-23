import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


class MultiLinearRegreesion(LinearRegression):
    def __init__(self):
        super(MultiLinearRegreesion, self).__init__()
        pass
    def fit(self, train_x, train_y):
        X, y = train_x, train_y
        P = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        params = np.dot(P, y)
        self.params = params


boston_datasets = load_boston()
X = boston_datasets.data
Y = boston_datasets.target


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.3)


model = MultiLinearRegreesion()
model.fit(train_x, train_y)


model.score(test_x, test_y)



