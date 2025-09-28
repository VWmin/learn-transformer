import random

import numpy as np


def sigmoid(x):
    """
    sigmoid 函数
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    """
    sigmoid 导函数
    """
    return sigmoid(x) * (1 - sigmoid(x))



class Network:
    def __init__(self, sizes):
        """
        描述每层有多少个神经元，例如 [2, 3, 1] 表示第一层2个，第二层3个，第三层1个
        [28*28, 16, 16, 10]
        该实现是数学的视角：x = wx + b
        """
        self.sizes = sizes
        self.layers = len(sizes)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # 权重是一个形状为 [y, x] 的矩阵
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 偏置项是一个形状为 [y, 1] 的列向量

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, train_data, epochs, batch_size, eta, test_data=None):
        """
        stochastic gradient descent，随机梯度下降算法
        """
        if test_data: n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            random.shuffle(train_data)
            # 一次从 train_data 中加载 batch_size 个数据
            batches = [train_data[k: k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete.")

    def update_batch(self, batch, eta):
        """
        eta: 学习率
        """
        batch_size = len(batch)
        # 偏导数
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for x, y in test_data
        ]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


