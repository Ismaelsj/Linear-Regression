import numpy as np

from colors import colors
from predict import predict

def accuracy(data, features, label, thetas):
    X = data[features]
    Y_true = data[label]
    Y_pred = [ predict(i, thetas)[0] for i in X ]
    
        # Coefficient of determination
    u = np.sum((Y_true - Y_pred) ** 2)
    v = np.sum((Y_true - np.mean(Y_true)) ** 2)
    pred = 1 - (u / v)
    print('Accuracy: ' + colors.OKBLUE + str(round(pred * 100, 2)) + colors.ENDC + ' %\n')