import numpy as np

def sigmoid(x):
  # f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)


def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
  """NeuralNetwork constructor
  A neural network with:
    - Two inputs (x1,x2)
    - A hidden layer with 2 neurons (h1, h2)
    - An output layer with 1 neuron (o1)
  Each neuron has the same weights and bias
  """
  def __init__(self): 
    # Weights. Controls the signal strength
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases. Additional set of weights with no required inputs  
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    
    self.list_epoch = []
    self.list_loss = []
    

  """Calculates the output of the neural network
  Returns:
    int: output from the neural network
  """
  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    """
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    """
    learn_rate = 0.01
    epochs = 2000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # Do a feedforward
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # Calculate partial derivatives.
        # Naming: d_L_d_w1 represents "partial L / partial w1"
        derivative_Lypred = -2 * (y_true - y_pred)

        # Neuron o1
        derivative_o1w5 = h1 * deriv_sigmoid(sum_o1)
        derivative_o1w6 = h2 * deriv_sigmoid(sum_o1)
        derivative_o1b3 = deriv_sigmoid(sum_o1)
        derivative_o1h1 = self.w5 * deriv_sigmoid(sum_o1)
        derivative_o1h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        derivative_h1w1 = x[0] * deriv_sigmoid(sum_h1)
        derivative_h1w2 = x[1] * deriv_sigmoid(sum_h1)
        derivative_h1b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        derivative_h2w3 = x[0] * deriv_sigmoid(sum_h2)
        derivative_h2w4 = x[1] * deriv_sigmoid(sum_h2)
        derivative_h2b2 = deriv_sigmoid(sum_h2)

        # Update weights and bias
        # Neuron h1
        self.w1 -= learn_rate * derivative_Lypred * derivative_o1h1 * derivative_h1w1
        self.w2 -= learn_rate * derivative_Lypred * derivative_o1h1 * derivative_h1w2
        self.b1 -= learn_rate * derivative_Lypred * derivative_o1h1 * derivative_h1b1

        # Neuron h2
        self.w3 -= learn_rate * derivative_Lypred * derivative_o1h2 * derivative_h2w3
        self.w4 -= learn_rate * derivative_Lypred * derivative_o1h2 * derivative_h2w4
        self.b2 -= learn_rate * derivative_Lypred * derivative_o1h2 * derivative_h2b2

        # Neuron o1
        self.w5 -= learn_rate * derivative_Lypred * derivative_o1w5
        self.w6 -= learn_rate * derivative_Lypred * derivative_o1w6
        self.b3 -= learn_rate * derivative_Lypred * derivative_o1b3

      # Calculate loss
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        self.list_loss.append(loss)
        self.list_epoch.append(epoch)
        print("Epoch %d loss: %.3f" % (epoch, loss))