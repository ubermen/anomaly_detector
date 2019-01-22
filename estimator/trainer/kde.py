import seaborn.apionly as sns
import matplotlib.pyplot as plt
import numpy as np


class KDE_Model(object) :
  def __init__(self, bandwidth) :
    self.bandwidth = bandwidth

  def train(self, data):
    bandwidth = self.bandwidth
    ax = sns.kdeplot(data, bw=bandwidth, label="bw: " + str(bandwidth), shade=True)

    plt.legend()
    self.data_x, self.data_y = ax.lines[0].get_data()

  def test(self, x):
    y = []
    for xi in x:
        y.append(np.interp(xi, self.data_x, self.data_y))
    return y