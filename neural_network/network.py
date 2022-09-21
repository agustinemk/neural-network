from .layer import Layer
from .loss import Loss


class Network:
    def __init__(self, loss: Loss) -> None:
        self.layers: list[Layer] = []
        self._loss = loss

    def predict(self, input_data):
        # # sample dimension first
        # samples = len(input_data)
        # result = []

        # # run network over all samples
        # for i in range(samples):
        #     # forward propagation
        #     output = input_data[i]
        #     for layer in self.layers:
        #         output = layer.forward_propagation(output)
        #     result.append(output)

        # return result

        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    def add_layers(self, *layers: Layer) -> None:
        if not self.layers:
            # TODO: Throw error if last layer output doesnt match new layer input
            ...

        self.layers += [layer for layer in layers]


    def fit(self, x_train, y_train, epochs, learning_rate, verbose=False):
        # Throw error if lenght of x, y doesnt match
        assert(len(x_train) == len(y_train))

        samples = len(x_train)

        # training
        for i in range(epochs):
            loss = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                if verbose: loss += self._loss.g(y_train[j], output)

                # backward propagation
                output_error = self._loss.g_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)


            if verbose:
                # calculate average loss on all samples
                loss /= samples
                print(f"epoch {i+1}/{epochs} {loss=}")
