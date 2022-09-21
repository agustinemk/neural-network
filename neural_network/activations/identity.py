from ..activation import Activation


class Identity(Activation):
    def g(self, x):
        return x

    def g_prime(self, x):
        return 1
