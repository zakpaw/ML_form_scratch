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

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yh = self.sigmoid(self.z3)
        return yh


if __name__ == '__main__':
    X, y = spiral_data(100, 3)
    nn = Neural_Network(input_l=2, output_l=3, hidden_l=2)
    print(nn.forward(X))
