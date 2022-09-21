from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def g(self, true_output, predicted_output):
        ...

    @abstractmethod
    def g_prime(self, true_output, predicted_output):
        ...
