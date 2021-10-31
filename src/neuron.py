import numpy as np

def sigmoid(x):
  # f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  """Neuron constructor"""
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  """Calculates the output of the neuron
  Returns:
		int: output from neuron
  """
  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)