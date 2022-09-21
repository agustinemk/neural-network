# Neural Network
## Introduction
This repository is a basic implementation from scratch of neural network (NNs) systems using only numpy library.

## Usage
```python
from neural_network import Network
from neural_network.activations import Tanh
from neural_network.losses import MSE

def load_data():
    ...
    return (x_train, y_train)

x_train, y_train = load_data()

network = Network(loss=MSE())
network.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
```

