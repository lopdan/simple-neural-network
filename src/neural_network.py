import numpy as np

from neuron import Neuron

class NeuralNetwork:
  """NeuralNetwork constructor
  A neural network with:
    - Two inputs (x1,x2)
    - A hidden layer with 2 neurons (h1, h2)
    - An output layer with 1 neuron (o1)
  Each neuron has the same weights and bias
  """
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  """Calculates the output of the neural network
  Returns:
		int: output from the neural network
  """
  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1