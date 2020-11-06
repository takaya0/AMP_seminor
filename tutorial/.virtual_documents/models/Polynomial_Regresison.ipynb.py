import numpy as np
from matplotlib import pyplot as plt


class Polynomial_Regresison():
    def __init__(self, degree):
        self.degree = degree
    
    def __call__(self, x):
        return self.forward(x)
    def fit(self, x_train, t_train, alpha=0):
        X = np.ones((len(x_train), self.degree + 1))
        for i in range(len(x_train)):
            for k in range(1, self.degree + 1):
                X[i][k] = np.power(x_train[i], k)
        regular = alpha * np.eye(int(self.degree + 1))
        W = np.dot(np.linalg.inv(np.dot(X.T, X) + regular), X.T)
        W = np.dot(W, t_train)
        self.W = W
    def forward(self, x):
        res = np.poly1d(self.W[::-1])
        return res(x)


def get_data(data_size, noise = False):
    x = sorted(np.random.randint(-30, 31, data_size))
    res = np.poly1d([0.6, -10, -5, 0.6])
    y = res(x)
    if noise:
        y = y + np.random.randint(-200, 201, data_size)
    return x, y


train_x, train_t = get_data(10, noise = True)


model = Polynomial_Regresison(degree=3)


model.fit(train_x, train_t)


res = np.poly1d([1, -10, -5, 0.6])
domain = np.arange(-30, 31, 0.1)
true_output = np.array([ res(x) for x in domain])
plt.scatter(train_x, train_t)
plt.plot(domain, model(domain), label='model output')
plt.plot(domain, true_output, 'r', label = 'real output')
plt.legend()
print(train_x)
print(model(train_x))
plt.savefig('overfitting.png')


print('model output: {}'.format(model(13)))
print('real output: {}'.format(res(13)))


model.fit(train_x, train_t, alpha=8_000_000)


res = np.poly1d([1, -10, -5, 0.6])
domain = np.arange(-30, 31, 0.1)
true_output = np.array([ res(x) for x in domain])
plt.scatter(train_x, train_t)
plt.plot(domain, model(domain), label='model output')
plt.plot(domain, true_output, 'r', label = 'real output')
plt.legend()
print(train_x)
print(model(train_x))
plt.savefig('regulared.png')


print('model output: {}'.format(model(13)))
print('real output: {}'.format(res(13)))



