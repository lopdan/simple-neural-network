import numpy as np
from neuron import Neuron
from neural_network import NeuralNetwork


def main():
	network = NeuralNetwork()
	x = np.array([2, 3])
	print(network.feedforward(x))
 
if __name__ == "__main__":
	main()