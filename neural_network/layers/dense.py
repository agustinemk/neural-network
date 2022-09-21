from ..layer import Layer
from ..activation import Activation
import numpy as np


class Dense(Layer):
    def __init__(
        self, input_size: int, output_size: int, activation: Activation
    ) -> None:
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
