from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def g(self, x):
        ...

    @abstractmethod
    def g_prime(self, x):
        ...
