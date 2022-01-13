import numpy as np
from BaseLayer import BaseLayer

class Dense(BaseLayer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        #print("dense out", np.dot(self.weights, self.input) + self.bias)
        return np.dot(self.weights, self.input) + self.bias #matrix multiplication

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient) #.T = transposed
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient