import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt

sys.path.append('./utils')
from scaling import scaling
from cost import cost
from train import fit_with_cost
from predict import predict
from visualize import visualizeRegression
from accuracy import accuracy
from colors import colors
import params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=params.ALPHA, help="Set alpha")
    parser.add_argument("-i" ,"--nbItr", type=int, default=params.NB_ITERATIONS, help="Set number of iterations")
    parser.add_argument("-f" ,"--feature", type=str, default=params.FEATURES, help="Set feature name")
    parser.add_argument("-l" ,"--label", type=str, default=params.LABELS, help="Set label name")
    parser.add_argument("-c" ,"--cost", action='store_true', help="Cost visu")
    parser.add_argument("-s" ,"--accuracyScore", action='store_true', help="Accuracy score")
    parser.add_argument("-p" ,"--dataPath", type=str, default=(params.DIR_PATH + params.DATA_PATH + params.DATA_NAME), help="Data absolute path")
    args = parser.parse_args()

    alpha = args.alpha 
    nbItr = args.nbItr
    featureName = args.feature
    labelName = args.label
    dataPath = args.dataPath
    accuracyScore = args.accuracyScore

        # get data
    try:
        data = pd.read_csv(dataPath)
    except FileNotFoundError:
        print(colors.FAIL + 'Data not found.' + colors.ENDC)
        exit(0)

        # get features and labels
    try:
        x = np.array(data[featureName])
        y = np.array(data[labelName])
    except KeyError:
        print(colors.FAIL + 'Wrong features or labels\nPlease use [-f/--feature] FEATURE and/or [-l/--label] LABEL' + colors.ENDC)
        exit(0)

    print('Parameters :')
    print('  alpha :', alpha)
    print('  iterations :', nbItr)
    print('  accuracyScore :', accuracyScore, '\n')

        # create thetas
    theta = np.array([[0], [0]], float)

        # scale input
    scaledX = scaling(x)

        # train
    theta, cost_history = fit_with_cost(scaledX, y, theta, alpha, nbItr)

    if args.cost:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(np.arange(len(cost_history)), cost_history)
        ax.set(xlabel='number of iterations', ylabel='cost',
            title='Cost')
        plt.show()

        # scaling
    theta[1] = theta[1] / (np.amax(x) - np.amin(x))

    print('Thetas :', theta, '\n')
    
    try:
        with open(params.DIR_PATH + params.JSON_PATH + params.JSON_NAME, 'w') as f:
            json.dump(theta.tolist(), f)
    except FileNotFoundError:
        print(colors.FAIL + "'" + params.DIR_PATH + params.JSON_PATH + "'" + ' directory not found.' + colors.ENDC)
        exit(0)


    if accuracyScore:
        accuracy(data, featureName, labelName, theta)
    
    visualizeRegression(theta, x, y, featureName, labelName)

    print(colors.OKGREEN + 'Thetas writtent in :', params.DIR_PATH + params.JSON_PATH + params.JSON_NAME + colors.ENDC)

if __name__ == "__main__":
    main()