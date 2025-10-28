# PyTorch-Inspired Autograd Engine & Neural Network Library

This project is a minimal, from-scratch implementation of an automatic differentiation (autodiff) engine and a simple neural network library in Python, heavily inspired by the mechanics of PyTorch.

The primary goal is to demystify backpropagation and the internal workings of modern deep-learning frameworks by building one from the ground up.

## Project Components

The project is split into two main files:

1.  **`backprop.py`**: This is the core autograd engine. It provides a `Scalar` class that tracks a single numerical value, its gradient, and the computational graph of operations that created it.
2.  **`nn.py`**: This file uses the `Scalar` class from `backprop.py` to build the components of a neural network: `Neuron`, `Layer`, and `Model` (a Multi-Layer Perceptron).

---

## `backprop.py` - The Autograd Engine

This file defines the `Scalar` class, which is the foundation of the entire project.

### The `Scalar` Class

A `Scalar` object is a wrapper around a single floating-point number (`data`). Crucially, it also holds:

* `self.grad`: The gradient of the final output (e.g., loss) with respect to this `Scalar`'s `data`. It's initialized to `0.0`.
* `self.prev`: A tuple of the "children" `Scalar` objects that were used to create this one. This builds the computational graph.
* `self.back`: A function that implements the **chain rule** for the *specific operation* that created this `Scalar`. It knows how to propagate the gradient backward to its children.

### How it Works: The Computational Graph

When you perform operations on `Scalar` objects, you are dynamically building a computational graph.

```python
a = Scalar(2.0)
b = Scalar(3.0)
c = a * b  # c stores (a, b) in self.prev
d = c + Scalar(5.0) # d stores (c, Scalar(5.0)) in self.prev
```
The graph looks like this: (a, b) -> c -> d

Each operation (like __add__ or __mul__) defines its own local back() function. For example, in __mul__:
```python
def back():
    # Chain rule for multiplication:
    # dL/da = dL/dc * dc/da = out.grad * other.data
    # dL/db = dL/dc * dc/db = out.grad * self.data
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
out.back = back
```


b = Scalar(3.0)
c = a * b  # c stores (a, b) in self.prev
d = c + Scalar(5.0) # d stores (c, Scalar(5.0)) in self.prev

The backwards() method is called on the final Scalar (usually the loss). It performs a topological sort on the computational graph to get all nodes in order.

Then, it iterates through the nodes in reverse order and calls the back() function for each one. This propagates the gradient from the output all the way back to the input nodes, perfectly applying the chain rule.

Supported Operations

    Addition (+)

    Multiplication (*)

    Subtraction (-)

    Division (/)

    Power (**)

    ReLU (.relu())

    Sigmoid (.sigmoid())
# Neural Network Library (`nn.py`)

This file provides the classes to construct a Multi-Layer Perceptron (MLP) using the `Scalar` autograd engine from `backprop.py`. It defines the hierarchical structure of a neural network: `Model` > `Layer` > `Neuron`.

**Dependency:** This module requires `backprop.py` to be in the same directory to import the `Scalar` class.

---

##  Core Classes

### `Neuron`

A single neuron, which is the smallest computational unit of the network.

```python
class Neuron:
    def __init__(self, nin, function = None):
        # ...
    
    def __call__(self, x):
        # ...

    def params(self):
        # ...
```
__init__(self, nin, function=None):

        nin: The number of input features (and therefore, the number of weights).

        function: A string ('relu' or 'sigmoid') specifying the activation function. If None, no activation is applied (linear).

        Initializes nin Scalar weights (randomly) and one Scalar bias (at 0.0).

    __call__(self, x):

        Performs the core neuron computation: sum(wi*xi) + b.

        x is expected to be a list of Scalar objects.

        The result is passed through the specified activation function.

        Returns a single Scalar object.

    params(self):

        Returns a list of all parameters (weights and bias) for this neuron.

### `Layer`

A layer of neurons that all receive the same input.
```python
class Layer:
    def __init__(self, nin, nout, function = None):
        # ...
    
    def __call__(self, x):
        # ...
    
    def params(self):
        # ...
```
__init__(self, nin, nout, function=None):

        nin: The number of input features (passed to each neuron).

        nout: The number of neurons in this layer.

        function: The activation function to be used by all neurons in this layer.

        Creates a list of nout Neuron objects.

    __call__(self, x):

        Passes the input x to every Neuron in the layer.

        Returns a list of Scalar objects (the outputs of all neurons).

    params(self):

        Collects and returns all parameters from all Neurons in this layer.

### `Model`

The complete Multi-Layer Perceptron (MLP) model.
```python
class Model:
    def __init__(self, nin, nouts, functions):
        # ...

    def __call__(self, x):
        # ...
    
    def params(self):
        # ...
```
__init__(self, nin, nouts, functions):

    nin: The number of input features for the first layer.

    nouts: A list of output sizes for each subsequent layer. The length of this list determines the number of layers.

    functions: A list of activation function names (e.g., ['relu', 'relu', None]) for each layer.

    Creates a list of Layer objects, chaining the inputs and outputs together.

__call__(self, x):

    Performs a full forward pass, passing the input x sequentially through each Layer.

    The output of one layer becomes the input to the next.

    Returns the final output of the last layer (a list of Scalars).

params(self):

    Collects and returns all parameters from all Layers, providing a single list of every trainable parameter in the entire model.
