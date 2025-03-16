import math as m
import random
from backprop import Scalar
class Neuron:
    def __init__(self, nin, function = None):
        self.w = [Scalar(random.uniform(-1,1)) for i in range(nin)]
        self.b = Scalar(0.0)
        self.function = function

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.function == 'relu':
            return act.relu()
        elif self.function == 'sigmoid':
            return act.sigmoid()
        else:
            return act

    def params(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, function = None):
        self.neurons = [Neuron(nin, function) for i in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()]


class Model:
    def __init__(self, nin, nouts, functions):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1], functions[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

