import numpy as np
from neural_network import NeuralNetwork
from plot import Plot


data = np.array([
  [-2, -1],
  [25, 6], 
  [17, 4], 
  [-15, -6],
  [28, 5],
  [-4, -3]
])
all_y_trues = np.array([
  1,
  0,
  0, 
  1,
  0,
  1, 
])

def main():
  network = NeuralNetwork()
  network.train(data, all_y_trues)
  plot = Plot()
  plot.print_plot(network.list_loss, network.list_epoch)
 
if __name__ == "__main__":
  main()