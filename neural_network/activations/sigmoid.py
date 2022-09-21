from ..activation import Activation
import numpy as np


class Sigmoid(Activation):
    def g(self, x):
        return 1 / (1 + np.exp(-x))

    def g_prime(self, x):
        return self.g(x) * (1 - self.g(x))
