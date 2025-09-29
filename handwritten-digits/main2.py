import random

import numpy as np
from networkx.algorithms.flow import minimum_cut


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
        """
        前向传播，得出模型预测值
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """
        stochastic gradient descent，随机梯度下降算法
        分批逐次进行梯度更新
        """
        if test_data: n_test = len(test_data)
        n = len(train_data)
        # 每个 epoch 意味着模型完整看过一遍训练集
        for j in range(epochs):
            # 保证每个batch的样本近似服从数据分布的独立随机采样
            random.shuffle(train_data)
            # 一次从 train_data 中加载 batch_size 个数据，构成一个 mini_batch
            mini_batches = [train_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 在现代化架构中，这里可以并行
            for mini_batch in mini_batches:
                # 在 mini_batch 上进行一次梯度更新
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete.")

    def update_mini_batch(self, mini_batch, eta):
        """
        eta: 学习率
        """
        mini_batch_size = len(mini_batch)
        # 放置偏导数结果
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 对于 batch 中的每个样本
        for x, y in mini_batch:
            # 反向传播计算样本梯度
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            # 每个样本的梯度相加
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # 梯度下降 -\eta\nablaC，不要忘了对batch内所有样本的损失求平均
        # 更新到所有参数
        self.weights = [w - (eta / mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        反向传播计算梯度
        x: 样本
        y: 样本标签
        """
        # 初始化梯度存储
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x  # 当前层的激活值，假设输入是-1层，显然这层的激活值就是输入
        activations = [x]  # 存储每层的激活值，用于反向传播
        zs = []  # 存储每层加权输入，对应每层前向传播后的结果，但还没有应用 sigmoid

        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b  # 前向传播计算加权和
            zs.append(z)
            activation = sigmoid(z)  # 加权和转为激活值
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # 输出层误差 * 多元函数偏导的链式法则
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 权重梯度
        nabla_b[-1] = delta  # 偏置项梯度

        # 隐藏层反向传播（从倒数第二层往后传播），计算过程同上
        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
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
