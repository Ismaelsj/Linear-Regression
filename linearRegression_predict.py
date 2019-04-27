import sys
import json
import argparse
import numpy as np

sys.path.append('./utils')
from predict import predict
from colors import colors
import params

def main():
    try:
        with open(params.DIR_PATH + params.JSON_PATH + params.JSON_NAME, 'r') as f:
            load = json.load(f)
            theta = np.array(load)
    except FileNotFoundError:
        print(colors.FAIL + params.DIR_PATH + params.JSON_PATH + params.JSON_NAME + ' not found.' + colors.ENDC)
        print("Please run " + colors.FAIL + "'linearRegression_train.py'" + colors.ENDC + " to create Thetas")
        exit(0)

    msg = "Please enter a mile age: "
    while True:
        try:
            mileage = int(input(msg))
            if mileage > 1000000 or mileage < 0:
                raise ValueError('Invalide number')
            break
        except ValueError:
            msg = "Please enter a number between 0 and 1 000 000: "
        
    print('prediction with a mile age of ' + colors.OKBLUE + str(mileage) + colors.ENDC + ' km : ' + colors.OKBLUE + str(round(predict(mileage, theta)[0], 2)) + colors.ENDC )


if __name__ == "__main__":
    main()