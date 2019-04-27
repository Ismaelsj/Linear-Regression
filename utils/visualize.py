import matplotlib.pyplot as plt
import numpy as np

def visualizeRegression(theta, x, y, xlabel, ylabel):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([-2, 300000])
    ax.set_ylim([0, 10000])
    ax.scatter(x, y)
    line_x = np.linspace(-2,300000, 20)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y, '--', color = 'red')
    ax.set(xlabel=xlabel, ylabel=ylabel,
            title='Data')
    plt.show()