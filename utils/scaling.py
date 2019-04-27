import numpy as np

def scaling(x):
    new_x = []
    x_max = np.amax(x)
    x_min = np.amin(x)
    for i in range(len(x)):
        new_x.append(x[i] / (x_max - x_min))
    return (new_x)