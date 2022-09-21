from ..activation import Activation
import numpy as np


class Tanh(Activation):
    def g(self, x):
        return np.tanh(x)

    def g_prime(self, x):
        return 1 - self.g(x) ** 2
