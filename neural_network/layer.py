from abc import abstractmethod


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
