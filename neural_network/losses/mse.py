from ..loss import Loss
import numpy as np


class MSE(Loss):
    def g(self, true_output, predicted_output):
        return np.mean(np.power(true_output - predicted_output, 2))

    def g_prime(self, true_output, predicted_output):
        return 2 * (predicted_output - true_output) / true_output.size
