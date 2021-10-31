import matplotlib.pyplot as plt

class Plot:
    def print_plot(self, x_axis, y_axis):
        plt.plot(x_axis, y_axis)
        plt.title("Training")
        plt.ylabel("Epochs")
        plt.xlabel("Loss")
        plt.legend(['Loss'], loc='upper right')
        plt.show()