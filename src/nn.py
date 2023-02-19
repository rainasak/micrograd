import random

from micrograd.src.engine import Value


class Module:
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	def parameters(self):
		return []

class Neuron(Module):
	def __init__(self, nin, nonlin = True):
		if nin is not int:
			raise TypeError('Number of inputs to neuron must be an integer')
		self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
		self.b = Value(0)
		self.nonlin = nonlin

	def __call__(self, x):
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		return act.relu() if self.nonlin else act

	def parameters(self):
		return [self.b] + self.w

	def __repr__(self) -> str:
		return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

class Layer(Module):
	def __init__(self, nin, nout, **kwargs) -> None:
		self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

	def __call__(self, x):
		out = [neuron(x) for neuron in self.neurons]
		return out[0] if len(out) == 1 else out

	def parameters(self):
		return [param for neuron in self.neurons for param in neuron.parameters()]

	def __repr__(self):
		return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"

class MLP(Module):
	def __init__(self, nin, nouts):
		sizes = [nin]+nouts
		self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i!=len(nouts)-1) for i in range(nouts)]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [param for layer in self.layers for param in layer.parameters()]

	def __repr__(self):
		return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"