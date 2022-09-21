import numpy as np
from abc import abstractmethod
from .activation import Activation


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input):
        # computes the output Y of a layer for a given input X
        ...

    @abstractmethod
    def backward_propagation(self, output_gradient, learning_rate):
        # computes dE/dX for a given dE/dY (and update parameters if any)
        ...


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, activation: Activation) -> None:
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size, 1)
        self._activation = activation

    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        # self.output = self.weights @ input + self.biases
        return self._activation.g(self.weights @ input + self.biases)

    def backward_propagation(self, output_gradient, learning_rate) -> np.ndarray:
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient

        # update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        return np.multiply(input_gradient, self._activation.g_prime(self.input))


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagation(self, input):
        return input.reshape(self.output_shape)

    def backward_propagation(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
