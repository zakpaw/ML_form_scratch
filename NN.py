import numpy as np
from spiral_data import spiral_data
np.random.seed(420)


class Layer_Dense:
    def __init__(self, inputs_count, neurons_count):
        # weights already transposed
        self.weights = np.random.uniform(-1, 1, (inputs_count, neurons_count))
        self.biases = np.zeros((1, neurons_count))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


if __name__ == '__main__':
    X, y = spiral_data(100, 3)

    l1 = Layer_Dense(len(X[0]), 5)
    activ1 = Activation_ReLU()

    l1.forward(X)
    activ1.forward(l1.output)
    print(activ1.output)
