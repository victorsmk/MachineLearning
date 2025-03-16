#PyTorch inspired backprop implementation
import math as m
class Scalar:

    def __init__(self, data, children = ()):
        self.data = data
        self.grad = 0.0
        self.prev = children
        self.back = lambda: None

    def relu(self):
        out = Scalar(max(self.data, 0.0), (self,))

        def back():
            self.grad += (0.0 if self.data < 0 else 1.0) * out.grad
        out.back = back
        return out

    def sigmoid(self):
        out = Scalar(1/(1+m.exp(1)**(-self.data)), (self,))

        def back():
            self.grad += out.grad * out.data * (1 - out.data)
        out.back = back
        return out

    def __repr__(self):
        return f"Scalar(data = {self.data}, grad = {self.grad})"

    def __add__(self, other):
        out = Scalar(self.data + other.data, (self, other))

        def back():
            self.grad += out.grad
            other.grad += out.grad
        out.back = back
        return out

    def __mul__(self, other):
        out = Scalar(self.data * other.data, (self, other))

        def back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.back = back
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        assert isinstance(other.data, (int, float)), "Please enter an integer/float power"
        out = Scalar(self.data ** other.data, (self, other))

        def back():
            self.grad += other.data * (self.data**(other.data - 1)) * out.grad
            other.grad += out.data * m.log(self.data, m.exp(1)) * out.grad
        out.back = back
        return out

    def backwards(self):
        nodes = []
        visited = set()
        def build_list(current_node):
            if current_node not in visited:
                visited.add(current_node)
                for child in current_node.prev:
                    build_list(child)
                nodes.append(current_node)
        build_list(self)

        self.grad = 1.0
        for node in reversed(nodes):
            node.back()

    def __neg__(self):
        return Scalar(self.data*(-1))

    def __truediv__(self, other):
        return self * (other**-1)

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1

