import math
from re import A


class Value:
	def __init__(self, data, _children=(), _op='', label='') -> None:
		self.data = data
		self.grad = 0.0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
		self.label = label

	def __repr__(self) -> str:
		return f'Value(data={self.data})'

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, (self, other), '+')

		def _backward():
			self.grad += 1.0 * out.grad
			other.grad += 1.0 * out.grad

		out._backward = _backward
		return out

	def __radd__(self, other):
		return self+other

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, (self, other), '*')

		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad

		out._backward = _backward
		return out

	def __pow__(self, other):
		assert isinstance(other, (int, float)), "only support int/float powers for now"
		out = Value(self.data**other, (self, ), f'**{other}')

		def _backward():
			self.grad += other*(self.data**(other-1))*out.grad

		out._backward = _backward
		return out

	def __rmul__(self, other):
		return self*other

	def __truediv__(self, other):
		return (self * other**-1)

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + (-other)

	def tanh(self):
		x = self.data
		t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
		out = Value(t, (self, ), 'tanh')

		def _backward():
			self.grad += (1-t**2) * out.grad

		out._backward = _backward
		return out

	def exp(self):
		x = self.data
		out = Value(math.exp(x), (self, ), 'exp')

		def _backward():
			self.grad += out.data * out.grad

		out._backward = _backward
		return out

	def relu(self):
		x = self.data
		out = Value(max(x, 0), (self, ), 'relu')

		def _backward():
			self.grad += (1 if out.data > 0 else 0) * out.grad

		out._backward = _backward
		return out

	def backward(self):
		visited = set()
		topSort = []

		def buildTopSort(node):
			if node not in visited:
				visited.add(node)
				for child in node._prev:
					buildTopSort(child)
				topSort.append(node)

		self.grad = 1
		buildTopSort(self)
		for node in reversed(topSort):
			node._backward()
