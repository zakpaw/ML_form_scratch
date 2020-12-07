import numpy as np
from spiral_data import spiral_data
np.random.seed(420)


class Neural_Network:
    def __init__(self, input_l, output_l, hidden_l):
        self.input_count = input_l
        self.output_count = output_l
        self.hidden_count = hidden_l

        self.W1 = np.random.randn(self.input_count, self.hidden_count)
        self.W2 = np.random.randn(self.hidden_count, self.output_count)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        exp = np.exp(-x)
        return exp/((1+exp)**2)

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yh = self.sigmoid(self.z3)
        return yh

    def cost_fun(self, X, y):
        self.yh = self.forward(X)
        return 0.5*sum((y-self.yh)**2)

    def cost_fun_prime(self, X, y):
        self.yh = self.forward(X)

        delta3 = np.multiply(-(y-self.yh), self.sigmoid_prime(self.z3))
        derivJW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        derivJW1 = np.dot(X.T, delta2)

        return derivJW1, derivJW2


if __name__ == '__main__':
    X, y = spiral_data(100, 3)
    nn = Neural_Network(input_l=2, output_l=3, hidden_l=2)
    print(y)
    print(nn.cost_fun_prime(X, y))
